"""LoRA trainer: EXPO-style base-policy fine-tuning, publishable to a live policy server.

Trains *only* LoRA adapters on top of a frozen base policy (PI05 by default) with
the policy's own flow-matching / denoising loss on a LeRobot dataset -- the same
imitation objective EXPO uses to keep fine-tuning its expressive base policy
online. There is no critic and no value weighting in this process: whatever
filtering you want (successful episodes only, corrected rollouts only) happens
upstream, in which episodes you point it at.

Every ``publish_freq`` steps it makes the new adapter (a few MB, never the base) the
one its AdapterService serves. A ``lora_policy_server.py`` pulls the latest adapter
over gRPC whenever it wants and swaps it into the running policy without a restart --
the trainer hosts the server and holds the current adapter; it never pushes.

Shortcut (installed by `pip install -e .`): ``ws-lora-finetuning`` -- the same
command runs on the workstation (bind ``--serve_host=127.0.0.1``, adapter over
loopback) or a remote GPU box (bind ``--serve_host=0.0.0.0``).
``--publish-freq`` == ``--publish_freq``; ``--lora_variant small`` for a quick test.

    # single-shot: fine-tune once on a fixed dataset
    ws-lora-finetuning \
        --pretrained_path=outputs/ablation/board_insertion_ablation_head_pi05_delta_recomputed_stats_25k \
        --dataset_repo_id=chomeed/board_insertion_ablation_dagger \
        --serve_host=127.0.0.1 --serve_port=8090 --publish-freq 500 \
        --steps=20000 --batch_size=8 --lr=1e-4

    # DAgger service: rounds of (train 500 steps -> publish -> reset lr -> wait
    # for new dagger data), mixing baseline demos 50/50 with the intervention
    # transitions of every <task>_sirius_round* dataset found so far
    ws-lora-finetuning \
        --pretrained_path=outputs/ablation/board_insertion_ablation_head_pi05_delta_recomputed_stats_25k \
        --dagger_loop=true \
        --baseline_dataset_repo_id=chomeed/board_insertion_ablation_head \
        --baseline_dataset_root=/data/lerobot/board_insertion_ablation_head \
        --dagger_datasets_dir=/data/lerobot/dagger \
        --dataset_repo_id=unused-in-dagger-mode \
        --serve_host=127.0.0.1 --serve_port=8090 --publish-freq 500 --lr=1e-4

Equivalent module form: ``python -m lora_finetuning.common.trainer``.
"""

import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed

from .configs import PI05_LORA_TARGETS, PI05_POLICY_TYPES, LoRATrainerConfig

# gRPC installs pthread_atfork handlers that log "skipping fork() handlers" every time
# the DataLoader forks a worker. The workers never touch gRPC, so disable fork support.
# Must be set before the transport import pulls in grpc.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

from .transport import AdapterPublisher  # noqa: E402

logger = logging.getLogger("lora_trainer")


def _enable_gradient_checkpointing(policy) -> int:
    """Flip on PI05's checkpointing flag wherever it lives. Returns modules touched."""
    if hasattr(policy.config, "gradient_checkpointing"):
        policy.config.gradient_checkpointing = True
    touched = 0
    for module in policy.modules():
        if hasattr(module, "gradient_checkpointing_enabled"):
            module.gradient_checkpointing_enabled = True
            touched += 1
    return touched


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak LR."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_policy(cfg: LoRATrainerConfig):
    """Load the base policy exactly as the policy server does, then wrap it in LoRA.

    Returns ``(policy, peft_model)``. ``get_peft_model`` injects the adapter layers
    into ``policy`` *in place*, so ``policy`` is what you run forward/backward on and
    ``peft_model`` is the handle used to save adapter-only checkpoints.
    """
    policy_cls = get_policy_class(cfg.policy_type)
    policy = policy_cls.from_pretrained(cfg.pretrained_path)

    # wrap_with_peft refuses to run without this (it is what PEFT records as the
    # adapter's base_model_name_or_path), and from_pretrained does not always set it.
    if not policy.config.pretrained_path:
        policy.config.pretrained_path = str(cfg.pretrained_path)

    policy.to(cfg.device)

    if cfg.gradient_checkpointing:
        touched = _enable_gradient_checkpointing(policy)
        logger.info(f"Gradient checkpointing enabled on {touched} module(s)")

    if cfg.resume_adapter_path is not None:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(policy, cfg.resume_adapter_path, is_trainable=True)
        logger.info(f"Resumed adapter from {cfg.resume_adapter_path}")
    else:
        overrides = cfg.lora.to_peft_overrides()
        if "target_modules" not in overrides and cfg.policy_type in PI05_POLICY_TYPES:
            # See PI05_LORA_TARGETS: the policy's own default regex is stale and would
            # silently skip the time-conditioning MLPs.
            overrides["target_modules"] = PI05_LORA_TARGETS
        logger.info(f"LoRA target_modules: {overrides.get('target_modules', '<policy default>')}")

        # Freezes every base parameter, then injects LoRA layers.
        peft_model = policy.wrap_with_peft(peft_cli_overrides=overrides)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in policy.parameters())
    logger.info(
        f"Trainable: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M params "
        f"({100 * n_trainable / max(n_total, 1):.3f}%)"
    )
    if n_trainable == 0:
        raise RuntimeError(
            "No trainable parameters after LoRA wrapping. Check lora.target_modules against "
            f"the module names of policy type '{cfg.policy_type}'."
        )

    return policy, peft_model


def build_dataloader(cfg: LoRATrainerConfig, policy) -> DataLoader:
    """LeRobot dataset windowed into action chunks the policy's loss can consume."""
    meta = LeRobotDatasetMetadata(cfg.dataset_repo_id, root=cfg.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, meta)
    logger.info(f"delta_timestamps: {delta_timestamps}")

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        episodes=cfg.episodes,
        delta_timestamps=delta_timestamps,
    )
    logger.info(f"Dataset: {dataset.num_frames} frames / {dataset.num_episodes} episodes")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


# ── DAgger round loop ──────────────────────────────────────────────────────
def _discover_dagger_rounds(cfg: LoRATrainerConfig) -> list[tuple[str, Path]]:
    """The per-round dagger datasets present under ``dagger_datasets_dir``, sorted
    (e.g. board_insertion_sirius_round1, ..._round2). Returns (repo_id, root)
    pairs, repo_id = the directory name. Only dirs with a finalized info.json
    count, so a half-written round is ignored until it's a valid dataset."""
    parent = Path(cfg.dagger_datasets_dir)
    rounds = []
    if parent.exists():
        for d in sorted(parent.glob(cfg.dagger_dataset_glob)):
            if d.is_dir() and (d / "meta" / "info.json").exists():
                rounds.append((d.name, d))
    return rounds


def _dagger_total_episodes(cfg: LoRATrainerConfig) -> int:
    """Total episodes across every dagger round dir -- grows both when the
    current round gains episodes and when a new round appears."""
    total = 0
    for _, root in _discover_dagger_rounds(cfg):
        try:
            total += int(json.loads((root / "meta" / "info.json").read_text()).get("total_episodes", 0))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    return total


def build_dagger_dataloader(cfg: LoRATrainerConfig, policy) -> DataLoader:
    """SIRIUS-weighted loader over [baseline demos, dagger round 1, round 2, ...].

    Reuses ``lerobot_sirius.dataset.SIRIUSDataset``: the baseline demos are class
    ``demo`` (all frames) and every dagger round contributes ONLY its intervention
    transitions (``p_robot_max=0``, ``p_preintv=0``), mixed at
    ``p_intv=dagger_sampling_ratio`` (0.5 -> 50/50 demo/intervention). Must be
    rebuilt every round because a new round dataset appears (and the current one
    grows) between rounds.

    ``use_recomputed_stats=False`` freezes SIRIUS's stats to the baseline demo
    dataset; and the training loop normalizes with the checkpoint's preprocessor
    regardless, so baseline normalization stats are never recomputed."""
    from lerobot_sirius.dataset import SIRIUSDataset

    meta = LeRobotDatasetMetadata(cfg.baseline_dataset_repo_id, root=cfg.baseline_dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, meta)
    rounds = _discover_dagger_rounds(cfg)
    # baseline (demo) first, then dagger rounds in order -- SIRIUS's curriculum
    # indexing expects the demo dataset(s) and the daggers in round order.
    repo_ids = [cfg.baseline_dataset_repo_id, *[rid for rid, _ in rounds]]
    roots = {cfg.baseline_dataset_repo_id: cfg.baseline_dataset_root}
    roots.update({rid: str(root) for rid, root in rounds})
    ds = SIRIUSDataset(
        repo_ids=repo_ids,
        roots=roots,
        p_intv=cfg.dagger_sampling_ratio,
        p_preintv=0.0,
        p_robot_max=0.0,  # only the intervention transitions from the dagger data
        use_recomputed_stats=False,  # freeze normalization to the baseline demo stats
        delta_timestamps=delta_timestamps,
    )
    logger.info(f"dagger mix ({len(rounds)} round(s)):\n{ds}")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=ds.make_sampler(),
        num_workers=cfg.num_workers,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _run_dagger_loop(cfg, policy, peft_model, preprocessor, publisher, autocast, version: int) -> int:
    """Round loop: wait for new dagger episodes -> rebuild the 50/50 mixed loader
    -> reset LR + scheduler -> train ``publish_freq`` steps -> publish -> repeat.
    ``cfg.steps <= 0`` runs forever (until Ctrl-C)."""
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    unbounded = cfg.steps <= 0
    global_step = 0
    trained_through = 0  # dagger episode count the last round trained on
    round_idx = 0
    policy.train()

    while unbounded or global_step < cfg.steps:
        # 1. wait for the converter to add new dagger episodes / a new round.
        have = _dagger_total_episodes(cfg)
        announced = False
        while have <= trained_through or have == 0:
            if not announced:
                logger.info(
                    f"round {round_idx}: waiting for new dagger episodes "
                    f"(have {have}, trained through {trained_through})..."
                )
                announced = True
            time.sleep(cfg.wait_poll_s)
            have = _dagger_total_episodes(cfg)

        # 2. rebuild the mixed loader over baseline + all dagger rounds so far.
        try:
            dataloader = build_dagger_dataloader(cfg, policy)
        except ValueError as e:
            # e.g. the new episodes carry no intervention frames yet, so the
            # intv class is empty and p_intv can't be satisfied -- wait for more.
            logger.info(f"round {round_idx}: {e}; waiting for more dagger data")
            trained_through = have
            time.sleep(cfg.wait_poll_s)
            continue

        # 3. reset LR + scheduler each round (fresh warmup over the round).
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
        if cfg.reset_optimizer_each_round:
            optimizer = torch.optim.AdamW(
                [p for p in policy.parameters() if p.requires_grad],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        round_len = cfg.publish_freq if unbounded else min(cfg.publish_freq, cfg.steps - global_step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: _lr_lambda(s, cfg.warmup_steps, round_len)
        )

        # 4. train one round.
        dl_iter = iter(dataloader)
        disp_loss, disp_n = 0.0, 0  # for the progress bar (reset each log)
        round_loss, round_n = 0.0, 0  # for the published loss (whole round)
        pbar = tqdm(total=round_len, desc=f"round {round_idx} -> v{version}", unit="step", dynamic_ncols=True)
        for _ in range(round_len):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(dataloader)
                    batch = next(dl_iter)
                batch = preprocessor(batch)
                with autocast:
                    loss, _ = policy.forward(batch)
                    loss = loss / cfg.grad_accum_steps
                loss.backward()
                full = loss.item() * cfg.grad_accum_steps
                disp_loss += full
                disp_n += 1
                round_loss += full
                round_n += 1
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
                error_if_nonfinite=False,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            pbar.update(1)
            if global_step % cfg.log_freq == 0:
                pbar.set_postfix(
                    step=f"{global_step}",
                    loss=f"{disp_loss / max(disp_n, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}",
                )
                disp_loss, disp_n = 0.0, 0
        pbar.close()

        # 5. publish this round's adapter, then loop back to wait for more data.
        avg = round_loss / round_n if round_n else None
        meta = publisher.publish(peft_model, version=version, step=global_step, loss=avg)
        logger.info(
            f"round {round_idx}: published adapter v{meta.version} "
            f"(step {global_step}, {have} dagger episode(s), loss "
            f"{avg:.4f})" if avg is not None else f"round {round_idx}: published v{meta.version}"
        )
        version += 1
        trained_through = have
        round_idx += 1

    return version


@draccus.wrap()
def train(cfg: LoRATrainerConfig):
    # force=True: lerobot's import installs a root WARNING handler; without force this
    # basicConfig is a no-op and all INFO training logs below are swallowed.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # Silence PI05's per-image resize_with_pad_torch warning (fires every forward).
    logging.getLogger("lerobot.policies.pi05.modeling_pi05").setLevel(logging.ERROR)
    logger.info(pformat(asdict(cfg)))

    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    policy, peft_model = build_policy(cfg)

    # Same preprocessor the policy server runs, loaded from the same checkpoint, so
    # normalization stats and tokenization match at train and inference time.
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    publisher = AdapterPublisher(host=cfg.serve_host, port=cfg.serve_port)
    publisher.start()
    version = 1

    if cfg.publish_at_start:
        # LoRA initializes B=0, so this adapter is an exact identity. Serving it up
        # front lets the policy server inject the adapter layers once, on its first
        # pull during a quiet moment, and makes every later update a cheap weight swap.
        meta = publisher.publish(peft_model, version, step=0)
        logger.info(f"Published identity adapter v{meta.version}")
        version += 1

    autocast = torch.autocast(
        device_type="cuda" if cfg.device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=cfg.use_amp,
    )

    # DAgger service: train publish_freq steps on a 50/50 baseline+dagger mix
    # (dagger contributes only its intervention transitions), publish, reset the
    # LR + scheduler, then wait for the converter to add new dagger episodes /
    # a new round. Normalization always reuses the baseline checkpoint stats.
    if cfg.dagger_loop:
        try:
            _run_dagger_loop(cfg, policy, peft_model, preprocessor, publisher, autocast, version)
        except KeyboardInterrupt:
            logger.info("dagger loop interrupted")
        logger.info("dagger loop finished; still serving the latest adapter (Ctrl-C to exit)")
        try:
            publisher.wait()
        except KeyboardInterrupt:
            publisher.stop()
        return

    # ---- single-shot training on a fixed dataset ----
    dataloader = build_dataloader(cfg, policy)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, cfg.warmup_steps, cfg.steps)
    )

    policy.train()
    dl_iter = iter(dataloader)
    running_loss = 0.0
    loss_count = 0
    step_start = time.perf_counter()

    # One bar per publish cycle: fills 0 -> publish_freq, then resets on each publish
    # so you can see how far off the next adapter is. Overall step is in the postfix.
    def _cycle_total(done: int) -> int:
        return min(cfg.publish_freq, cfg.steps - done)

    pbar = tqdm(
        total=_cycle_total(0),
        desc=f"train -> v{version}",
        unit="step",
        dynamic_ncols=True,
    )

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dataloader)
                batch = next(dl_iter)

            batch = preprocessor(batch)

            with autocast:
                loss, _ = policy.forward(batch)
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            running_loss += loss.item() * cfg.grad_accum_steps
            loss_count += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )
        optimizer.step()
        scheduler.step()
        pbar.update(1)

        if step % cfg.log_freq == 0:
            avg_loss = running_loss / max(loss_count, 1)
            dt = (time.perf_counter() - step_start) / cfg.log_freq
            pbar.set_postfix(
                step=f"{step}/{cfg.steps}",
                loss=f"{avg_loss:.4f}",
                grad=f"{float(grad_norm):.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
                s_step=f"{dt:.2f}",
            )
            running_loss = 0.0
            loss_count = 0
            step_start = time.perf_counter()

        if step % cfg.publish_freq == 0:
            avg_loss = running_loss / loss_count if loss_count else None
            meta = publisher.publish(peft_model, version=version, step=step, loss=avg_loss)
            tqdm.write(f"Published adapter v{meta.version} (step {step})")
            version += 1
            if step < cfg.steps:  # start a fresh bar for the next publish cycle
                pbar.reset(total=_cycle_total(step))
                pbar.set_description(f"train -> v{version}")

    pbar.close()
    meta = publisher.publish(peft_model, version=version, step=cfg.steps)
    logger.info(f"Training done. Final adapter v{meta.version}")

    # Keep serving the final adapter so a policy server can still pull it after
    # training ends; Ctrl-C to exit.
    try:
        publisher.wait()
    except KeyboardInterrupt:
        publisher.stop()


if __name__ == "__main__":
    register_third_party_plugins()
    train()

"""LoRA learner: EXPO-style base-policy fine-tuning, publishable to a live policy server.

Trains *only* LoRA adapters on top of a frozen base policy (PI05 by default) with
the policy's own flow-matching / denoising loss on a LeRobot dataset -- the same
imitation objective EXPO uses to keep fine-tuning its expressive base policy
online. There is no critic and no value weighting in this process: whatever
filtering you want (successful episodes only, corrected rollouts only) happens
upstream, in which episodes you point it at.

Every ``publish_freq`` steps it writes the adapter (a few MB, never the base) into
``adapter_dir``, where ``lora_policy_server.py`` picks it up and swaps it into the
running policy without a restart.

Usage:
    python -m lora_finetuning.learner \
        --pretrained_path=/path/to/pi05_checkpoint/pretrained_model \
        --dataset_repo_id=chomeed/board_handover \
        --dataset_root=/path/to/dataset \
        --adapter_dir=/path/to/adapters \
        --steps=20000 --batch_size=8 --lr=1e-4
"""

import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed

from .adapter_store import next_version, publish_adapter
from .configs import PI05_LORA_TARGETS, PI05_POLICY_TYPES, LoRALearnerConfig

logger = logging.getLogger("lora_learner")


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


def build_policy(cfg: LoRALearnerConfig):
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


def build_dataloader(cfg: LoRALearnerConfig, policy) -> DataLoader:
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


@draccus.wrap()
def train(cfg: LoRALearnerConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
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

    dataloader = build_dataloader(cfg, policy)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, cfg.warmup_steps, cfg.steps)
    )

    adapter_dir = Path(cfg.adapter_dir).expanduser()
    version = next_version(adapter_dir)

    if cfg.publish_at_start:
        # LoRA initializes B=0, so this adapter is an exact identity. Publishing it
        # up front lets the server inject the adapter layers once, during a quiet
        # moment, and makes every later update a cheap in-place weight swap.
        meta = publish_adapter(peft_model, adapter_dir, version, step=0, keep_last=cfg.keep_last)
        logger.info(f"Published identity adapter v{meta.version} -> {meta.path(adapter_dir)}")
        version += 1

    autocast = torch.autocast(
        device_type="cuda" if cfg.device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=cfg.use_amp,
    )

    policy.train()
    dl_iter = iter(dataloader)
    running_loss = 0.0
    loss_count = 0
    step_start = time.perf_counter()

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

        if step % cfg.log_freq == 0:
            avg_loss = running_loss / max(loss_count, 1)
            dt = (time.perf_counter() - step_start) / cfg.log_freq
            logger.info(
                f"step {step}/{cfg.steps} | loss {avg_loss:.4f} | "
                f"grad_norm {float(grad_norm):.3f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                f"{dt:.3f}s/step"
            )
            running_loss = 0.0
            loss_count = 0
            step_start = time.perf_counter()

        if step % cfg.publish_freq == 0:
            avg_loss = running_loss / loss_count if loss_count else None
            meta = publish_adapter(
                peft_model,
                adapter_dir,
                version=version,
                step=step,
                loss=avg_loss,
                keep_last=cfg.keep_last,
            )
            logger.info(f"Published adapter v{meta.version} (step {step}) -> {meta.path(adapter_dir)}")
            version += 1

    meta = publish_adapter(
        peft_model, adapter_dir, version=version, step=cfg.steps, keep_last=cfg.keep_last
    )
    logger.info(f"Training done. Final adapter v{meta.version} -> {meta.path(adapter_dir)}")


if __name__ == "__main__":
    register_third_party_plugins()
    train()

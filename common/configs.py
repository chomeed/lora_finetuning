"""Draccus configs for the LoRA trainer and the adapter-aware policy server."""


from dataclasses import dataclass, field

from lerobot.async_inference.configs import PolicyServerConfig

# LoRA targets for PI05: the action expert's q/v attention projections, the action
# in/out projections, and the flow-matching time-conditioning MLPs. The SigLIP tower
# and the PaliGemma trunk are deliberately left frozen.
#
# We do NOT use PI05Policy._get_default_peft_targets() because it is stale: it targets
# `model.state_proj` (PI05 has no such module -- state enters as prefix tokens) and
# `model.action_time_mlp_{in,out}` (PI0's names; PI05 calls them `model.time_mlp_{in,out}`).
# Those two never match, so the stock default silently skips the time MLPs.
PI05_LORA_TARGETS = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj"
    r"|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
)

# Policy types for which PI05_LORA_TARGETS applies.
PI05_POLICY_TYPES = ("pi05", "pi05_mem")

# Shared default for the server's --version_status_file and the converter's
# --policy_status_file. Both processes run on the same workstation, so pointing
# them at the same path by default means every dagger manifest row gets its
# adapter_version/trainer_step tag with no extra flags. The server rewrites the
# file at startup (version 0 = base policy), so a stale file from a previous
# run cannot mislabel episodes.
DEFAULT_POLICY_STATUS_PATH = "/tmp/lora_policy_version.json"


@dataclass
class LoRASpec:
    """LoRA hyperparameters.

    ``target_modules=None`` resolves to ``PI05_LORA_TARGETS`` for PI05-family policies,
    and otherwise to the policy's own ``_get_default_peft_targets()``.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] | str | None = None
    init_lora_weights: bool | str = True

    def to_peft_overrides(self) -> dict:
        overrides = {
            "method_type": "lora",
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "init_lora_weights": self.init_lora_weights,
        }
        if self.target_modules is not None:
            overrides["target_modules"] = self.target_modules
        return overrides


# Predefined LoRA sizes. `--lora_variant small` is the quick-test config (fewer
# trainable params -> faster steps, less VRAM); `medium` matches the standalone
# default (r=16); `big` for a higher-capacity adapter. Override individual knobs
# with `--lora.r` / `--lora.lora_alpha` etc. after picking a variant.
LORA_VARIANTS = {
    "small": dict(r=4, lora_alpha=8),
    "medium": dict(r=16, lora_alpha=32),
    "big": dict(r=32, lora_alpha=64),
}


@dataclass
class LoRATrainerConfig:
    """Config for `trainer.py`: flow-matching BC on a LeRobot dataset, LoRA only."""

    # --- what to fine-tune -------------------------------------------------
    pretrained_path: str  # base policy checkpoint (the same one policy_server loads)
    dataset_repo_id: str  # LeRobot dataset id, e.g. "chomeed/board_handover"

    # --- adapter streaming (trainer hosts the gRPC server) -----------------
    serve_host: str = "0.0.0.0"  # interface the AdapterService binds
    serve_port: int = 8090  # port policy servers dial to subscribe

    dataset_root: str | None = None  # local dataset dir, if not under HF cache
    policy_type: str = "pi05"
    episodes: list[int] | None = None  # subset of episodes; None = all
    rename_map: dict[str, str] = field(default_factory=dict)

    # --- optimization ------------------------------------------------------
    device: str = "cuda"
    seed: int = 1000
    steps: int = 20_000
    batch_size: int = 8
    grad_accum_steps: int = 1
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 200
    grad_clip_norm: float = 1.0
    use_amp: bool = True  # bf16 autocast
    gradient_checkpointing: bool = True  # trades ~30% step time for a lot of VRAM

    lora: LoRASpec = field(default_factory=LoRASpec)
    # One of LORA_VARIANTS ("small"/"medium"/"big"); sets lora.r + lora.lora_alpha
    # so you can quick-test with `--lora_variant small`. None = use `lora` as given.
    lora_variant: str | None = None

    # --- publishing --------------------------------------------------------
    publish_freq: int = 500  # steps between adapter publishes
    publish_at_start: bool = True  # publish the (identity) adapter at step 0
    log_freq: int = 50
    resume_adapter_path: str | None = None  # continue from a published adapter dir

    # --- dagger round loop (the "lora finetuning service" behavior) --------
    # When enabled, training runs in rounds: train `publish_freq` steps on a
    # 50/50 mix of the baseline demos and the (growing) dagger dataset, publish
    # the adapter, reset the LR + scheduler, then WAIT for the converter to add
    # new dagger episodes before starting the next round. Normalization always
    # reuses the baseline checkpoint's stats (via the preprocessor loaded from
    # `pretrained_path`) -- dataset stats are never recomputed.
    dagger_loop: bool = False
    baseline_dataset_repo_id: str | None = None  # fixed baseline demos to mix with dagger data
    baseline_dataset_root: str | None = None
    # Parent directory holding the per-round dagger datasets, each a LeRobot
    # dataset written by the converter and named <task>_sirius_round<N>
    # (e.g. board_insertion_sirius_round1, board_insertion_sirius_round2). Every
    # round the loop rebuilds a SIRIUS mix of the baseline + all rounds present.
    dagger_datasets_dir: str | None = None
    dagger_dataset_glob: str = "*sirius_round*"  # which subdirs are dagger rounds
    dagger_sampling_ratio: float = 0.5  # P*(intervention) in the SIRIUS mix (baseline demos take the rest)
    # A training round starts only when the converter *finalizes* a dagger round
    # (not on every new episode), and the mix uses only completed rounds. The
    # round size is the converter's --num_demos; leave this None to auto-infer it
    # from disk (the size of rounds the converter already rolled past), or set it
    # explicitly to override the inference.
    dagger_round_size: int | None = None
    wait_poll_s: float = 10.0  # between rounds, poll this often for new dagger episodes/rounds
    reset_optimizer_each_round: bool = False  # also reset AdamW moments (lr+scheduler always reset)
    # `steps` is the total-step cap across all rounds; <= 0 means run rounds
    # forever (until Ctrl-C).

    def __post_init__(self):
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.publish_freq < 1:
            raise ValueError(f"publish_freq must be >= 1, got {self.publish_freq}")
        if self.lora_variant is not None:
            if self.lora_variant not in LORA_VARIANTS:
                raise ValueError(
                    f"unknown lora_variant {self.lora_variant!r}; choose one of {sorted(LORA_VARIANTS)}"
                )
            for k, v in LORA_VARIANTS[self.lora_variant].items():
                setattr(self.lora, k, v)
        if self.dagger_loop:
            if not self.baseline_dataset_repo_id:
                raise ValueError("dagger_loop=True needs --baseline_dataset_repo_id to mix against")
            if not self.dagger_datasets_dir:
                raise ValueError(
                    "dagger_loop=True needs --dagger_datasets_dir (parent of the "
                    "<task>_sirius_round* datasets)"
                )
            if not 0.0 < self.dagger_sampling_ratio < 1.0:
                raise ValueError(
                    f"dagger_sampling_ratio must be in (0, 1), got {self.dagger_sampling_ratio}"
                )
            if self.dagger_round_size is not None and self.dagger_round_size < 1:
                raise ValueError(
                    f"dagger_round_size must be >= 1 when set, got {self.dagger_round_size}"
                )


@dataclass
class RealtimeConverterConfig:
    """Config for `realtime_converter.py`: watch an ingest dir for recorded HDF5
    episodes and append each one to a growing LeRobot dataset, throttled so it
    never starves the co-located (high-priority) policy server."""

    # --- what to watch / where to write ------------------------------------
    ingest_dir: str  # directory the robot uploads finished episode_*.h5 into
    # SINGLE-dataset mode target (required unless round mode). In round mode
    # (--dagger_datasets_dir) both are unused -- the round datasets are named
    # <task>_sirius_round<N> under that dir with repo_id <repo_namespace>/<name>.
    dataset_repo_id: str = ""  # LeRobot dataset id to append into, e.g. "chomeed/board_handover"
    dataset_root: str = ""  # local dir for the growing dataset (created on first episode)

    glob: str = "episode_*.h5"  # which files in ingest_dir are episodes
    default_task: str = ""  # task string when an episode's `task` attr is empty
    task_filter: str | None = None  # if set, only convert episodes whose task matches
    fps: int = 30  # fallback fps when the episode carries none

    # --- observation/action schema -----------------------------------------
    # Project the full-rig recording down to a policy's I/O schema before
    # writing the dataset (see common/schema.py MODE_SCHEMAS). "full" keeps all
    # 41-D state / 19-D action and every camera; the reduced modes select the
    # channels + cameras the matching checkpoint was trained on.
    mode: str = "full"

    # --- dagger provenance -------------------------------------------------
    # Which policy produced the rollouts this run ingests. Recorded per episode
    # in the dataset's dagger manifest (meta/dagger_manifest.jsonl). An episode
    # that tags its own policy in an HDF5 attr (policy_id/policy_path/base_ckpt)
    # overrides this; otherwise every episode this run converts is attributed
    # to `dagger_policy`. Empty -> "unknown".
    dagger_policy: str = ""

    # Path of the LoRAPolicyServer's version status file (its
    # --version_status_file): each manifest row then also records the adapter
    # version + trainer step live at conversion time -- i.e. which policy
    # version the workstation is currently serving. Defaults to the same shared
    # path as the server so version tagging works with no flags when both run
    # on this machine. Set to "" to disable.
    policy_status_file: str | None = DEFAULT_POLICY_STATUS_PATH

    # --- sirius round rollover ---------------------------------------------
    # When `dagger_datasets_dir` is set, the converter runs in ROUND mode: it
    # writes into `<task>_sirius_round<N>` datasets under that parent dir and,
    # once a round reaches `num_demos` episodes, it finalizes that round and
    # opens the next one -- converting continuously, never stopping. The trainer's
    # `--dagger_datasets_dir` should point at the same parent. `repo_namespace`
    # is the HF namespace for each round's repo_id (dir name = <task>_sirius_round<N>).
    dagger_datasets_dir: str | None = None
    repo_namespace: str = "chomeed"

    # --- episode cap -------------------------------------------------------
    # ROUND mode: episodes per round before rolling to the next dataset.
    # SINGLE-dataset mode (no dagger_datasets_dir): stop and exit once the dataset
    # holds this many episodes total. None -> no cap (single mode runs forever).
    # Exposed as `--num_demos` on the console script.
    num_demos: int | None = None

    # --- scan / loop -------------------------------------------------------
    poll_interval_s: float = 2.0  # sleep between ingest-dir scans when idle
    min_age_s: float = 1.0  # ignore files touched more recently (belt-and-suspenders vs partial writes)
    run_once: bool = False  # convert the current backlog and exit (no watching)
    num_decode_workers: int = 4  # threads decoding JPEGs ahead of the writer

    # --- video encoding ----------------------------------------------------
    # Streaming encoding pipes each frame into the video encoder as it is added,
    # instead of writing every frame as a temp PNG and batch-encoding the whole
    # episode at save time. It removes the temp-image disk churn and spreads the
    # encode CPU across the episode rather than bursting it at the end.
    streaming_encoding: bool = True
    vcodec: str = "libsvtav1"  # AV1; needs system ffmpeg with the encoder built in
    encoder_queue_maxsize: int = 30  # max buffered frames per camera (streaming only)
    encoder_threads: int | None = None  # threads per encoder instance; None = auto

    # --- source-file disposition (the ingest dir IS the queue) -------------
    # No ledger: a file sitting in the ingest dir is pending; once handled it
    # leaves the scan set. On a successful conversion the .h5 (and its .sha256
    # sidecar) is deleted -- the robot keeps the raw recording -- unless
    # delete_on_success is False, in which case it is moved into converted_dir.
    # Files that fail or can't be converted are moved to failed_dir so they are
    # not retried on every scan.
    delete_on_success: bool = True
    converted_dir: str | None = None  # move here instead of deleting; None -> <ingest_dir>/converted
    failed_dir: str | None = None  # unconvertible files land here; None -> <ingest_dir>/failed
    verify_checksum: bool = True  # if a <name>.sha256 sidecar exists, verify before converting

    # --- throttling (stay out of the policy server's way) ------------------
    gpu_index: int = 0  # GPU whose utilization gates conversion
    gpu_util_pause: int = 60  # pause when utilization.gpu (%) is at/above this
    gpu_util_resume: int = 40  # resume once it drops to/below this (hysteresis)
    throttle_poll_s: float = 5.0  # how often to re-check the throttle signal while paused
    # Touch this file to pause conversion regardless of GPU; remove it to resume.
    # None -> <ingest_dir>/.pause
    pause_file: str | None = None
    nice: int = 15  # CPU niceness applied to this process (and its encode children)
    ionice_idle: bool = True  # best-effort `ionice -c3` so disk IO yields to the server

    def __post_init__(self):
        if self.gpu_util_resume > self.gpu_util_pause:
            raise ValueError(
                f"gpu_util_resume ({self.gpu_util_resume}) must be <= gpu_util_pause "
                f"({self.gpu_util_pause}) for the hysteresis to make sense"
            )
        from .schema import MODE_SCHEMAS, resolve_mode

        if resolve_mode(self.mode) not in MODE_SCHEMAS:
            raise ValueError(
                f"unknown mode {self.mode!r}; choose one of {sorted(MODE_SCHEMAS)} "
                "(see common/schema.py)"
            )
        if self.num_demos is not None and self.num_demos < 1:
            raise ValueError(f"num_demos must be >= 1 when set, got {self.num_demos}")
        if self.dagger_datasets_dir is not None:
            if self.num_demos is None:
                raise ValueError("round mode (--dagger_datasets_dir) needs --num_demos (episodes per round)")
            if not self.default_task:
                raise ValueError("round mode needs --default_task (used to name <task>_sirius_round<N>)")
        elif not (self.dataset_repo_id and self.dataset_root):
            raise ValueError(
                "single-dataset mode needs --dataset_repo_id and --dataset_root "
                "(or use --dagger_datasets_dir for sirius round mode)"
            )


@dataclass
class LoRAPolicyServerConfig(PolicyServerConfig):
    """PolicyServerConfig plus adapter hot-reload settings."""

    # Address of the trainer's AdapterService, "host:port". None disables
    # hot-reload entirely, making this server behave exactly like the stock
    # PolicyServer.
    adapter_addr: str | None = None

    # Local dir to materialize received adapters into. None -> a temp dir.
    adapter_cache_dir: str | None = None

    # The server writes the currently-serving adapter version (+ trainer step
    # and loss) to this JSON file at startup (version 0 = base policy) and on
    # every swap. The realtime converter reads it (its --policy_status_file) so
    # each dagger episode's manifest row records which policy version was live
    # when it was collected. Defaults to the same shared path as the converter
    # so tagging works with no flags. Set to "" to disable.
    version_status_file: str | None = DEFAULT_POLICY_STATUS_PATH

    # Default (False) keeps the console to lifecycle events: startup, adapter
    # swaps, warnings/errors. True restores the stock firehose -- per-observation
    # inference timings and the transport layer's per-stream chatter.
    verbose: bool = False

    # Where a newer adapter is allowed to be swapped in:
    #   "chunk"     -> at any action-chunk boundary (fastest propagation)
    #   "handshake" -> only when a client (re)connects, i.e. between episodes
    reload_on: str = "chunk"

    def __post_init__(self):
        super().__post_init__()
        if self.reload_on not in ("chunk", "handshake"):
            raise ValueError(f"reload_on must be 'chunk' or 'handshake', got {self.reload_on!r}")

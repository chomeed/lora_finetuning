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

    # --- publishing --------------------------------------------------------
    publish_freq: int = 500  # steps between adapter publishes
    publish_at_start: bool = True  # publish the (identity) adapter at step 0
    log_freq: int = 50
    resume_adapter_path: str | None = None  # continue from a published adapter dir

    def __post_init__(self):
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.publish_freq < 1:
            raise ValueError(f"publish_freq must be >= 1, got {self.publish_freq}")


@dataclass
class RealtimeConverterConfig:
    """Config for `realtime_converter.py`: watch an ingest dir for recorded HDF5
    episodes and append each one to a growing LeRobot dataset, throttled so it
    never starves the co-located (high-priority) policy server."""

    # --- what to watch / where to write ------------------------------------
    ingest_dir: str  # directory the robot uploads finished episode_*.h5 into
    dataset_repo_id: str  # LeRobot dataset id to append into, e.g. "chomeed/board_handover"
    dataset_root: str  # local dir for the growing dataset (created on first episode)

    glob: str = "episode_*.h5"  # which files in ingest_dir are episodes
    default_task: str = ""  # task string when an episode's `task` attr is empty
    task_filter: str | None = None  # if set, only convert episodes whose task matches
    fps: int = 30  # fallback fps when the episode carries none

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


@dataclass
class LoRAPolicyServerConfig(PolicyServerConfig):
    """PolicyServerConfig plus adapter hot-reload settings."""

    # Address of the trainer's AdapterService, "host:port". None disables
    # hot-reload entirely, making this server behave exactly like the stock
    # PolicyServer.
    adapter_addr: str | None = None

    # Local dir to materialize received adapters into. None -> a temp dir.
    adapter_cache_dir: str | None = None

    # Where a newer adapter is allowed to be swapped in:
    #   "chunk"     -> at any action-chunk boundary (fastest propagation)
    #   "handshake" -> only when a client (re)connects, i.e. between episodes
    reload_on: str = "chunk"

    def __post_init__(self):
        super().__post_init__()
        if self.reload_on not in ("chunk", "handshake"):
            raise ValueError(f"reload_on must be 'chunk' or 'handshake', got {self.reload_on!r}")

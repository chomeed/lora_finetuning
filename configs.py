"""Draccus configs for the LoRA learner and the adapter-aware policy server."""


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
class LoRALearnerConfig:
    """Config for `learner.py`: flow-matching BC on a LeRobot dataset, LoRA only."""

    # --- what to fine-tune -------------------------------------------------
    pretrained_path: str  # base policy checkpoint (the same one policy_server loads)
    dataset_repo_id: str  # LeRobot dataset id, e.g. "chomeed/board_handover"
    adapter_dir: str  # watch dir the policy server polls

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
    keep_last: int = 5  # versioned dirs to retain on disk
    log_freq: int = 50
    resume_adapter_path: str | None = None  # continue from a published adapter dir

    def __post_init__(self):
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.publish_freq < 1:
            raise ValueError(f"publish_freq must be >= 1, got {self.publish_freq}")


@dataclass
class LoRAPolicyServerConfig(PolicyServerConfig):
    """PolicyServerConfig plus adapter hot-reload settings."""

    # Directory the learner publishes adapters into. None disables hot-reload
    # entirely, making this server behave exactly like the stock PolicyServer.
    adapter_dir: str | None = None

    # Where a newer adapter is allowed to be swapped in:
    #   "chunk"     -> at any action-chunk boundary (fastest propagation)
    #   "handshake" -> only when a client (re)connects, i.e. between episodes
    reload_on: str = "chunk"

    def __post_init__(self):
        super().__post_init__()
        if self.reload_on not in ("chunk", "handshake"):
            raise ValueError(f"reload_on must be 'chunk' or 'handshake', got {self.reload_on!r}")

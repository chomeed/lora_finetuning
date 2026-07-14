"""On-the-fly LoRA fine-tuning of a base policy, served through lerobot's async inference stack.

Two processes, one directory between them:

    learner.py            --adapter_dir=DIR   trains LoRA adapters, publishes versions
    lora_policy_server.py --adapter_dir=DIR   serves the policy, hot-swaps adapters

See adapter_store.py for the (deliberately tiny) hand-off protocol.
"""

from .adapter_store import AdapterVersion, AdapterWatcher, publish_adapter, read_latest
from .configs import LoRALearnerConfig, LoRAPolicyServerConfig, LoRASpec

__all__ = [
    "AdapterVersion",
    "AdapterWatcher",
    "LoRALearnerConfig",
    "LoRAPolicyServerConfig",
    "LoRASpec",
    "publish_adapter",
    "read_latest",
]

"""On-the-fly LoRA fine-tuning of a base policy, served through lerobot's async inference stack.

Two processes, connected by gRPC (pull model — the client asks when it wants params):

    trainer.py            --serve_port=PORT     trains LoRA adapters, hosts the AdapterService
    lora_policy_server.py --adapter_addr=H:PORT serves the policy, pulls + hot-swaps adapters

See the transport/ subpackage for the hand-off protocol.
"""

from .configs import LoRAPolicyServerConfig, LoRASpec, LoRATrainerConfig, RealtimeConverterConfig
from .transport import AdapterApplier, AdapterClient, AdapterPublisher, AdapterVersion

__all__ = [
    "AdapterApplier",
    "AdapterClient",
    "AdapterPublisher",
    "AdapterVersion",
    "LoRATrainerConfig",
    "LoRAPolicyServerConfig",
    "LoRASpec",
    "RealtimeConverterConfig",
]

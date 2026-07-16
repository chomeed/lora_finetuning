"""Shared library for the LoRA fine-tuning stack.

Role-specific entrypoints live in ``workstation_pc/`` and ``robot_pc/``;
everything they have in common lives here: configs, the trainer/policy-server
cores, the HDF5 episode reader, the observation/action schema, the dagger
manifest, and the gRPC adapter transport (``common/transport``).
"""

from .configs import (
    LoRAPolicyServerConfig,
    LoRASpec,
    LoRATrainerConfig,
    RealtimeConverterConfig,
)
from .manifest import DaggerManifest, ManifestEntry

__all__ = [
    "DaggerManifest",
    "ManifestEntry",
    "LoRATrainerConfig",
    "LoRAPolicyServerConfig",
    "LoRASpec",
    "RealtimeConverterConfig",
]

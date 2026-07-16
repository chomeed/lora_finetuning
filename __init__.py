"""On-the-fly LoRA fine-tuning of a base policy, served through lerobot's async inference stack.

Organized by the machine each piece runs on:

    workstation_pc/  policy server + realtime HDF5->LeRobot converter + LoRA trainer
                     (the same trainer runs on a remote GPU box: bind serve_host=0.0.0.0)
    robot_pc/        pushes recorded demo episodes to the workstation ingest queue
    common/          shared library: configs, trainer/policy-server cores, schema,
                     manifest, and the gRPC adapter transport (common/transport)

See each folder's README and the top-level README for the data/adapter loop.
"""

from .common.configs import LoRAPolicyServerConfig, LoRASpec, LoRATrainerConfig, RealtimeConverterConfig
from .common.transport import AdapterApplier, AdapterClient, AdapterPublisher, AdapterVersion

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

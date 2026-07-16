"""gRPC transport for LoRA adapters (pull model): trainer (server) -> policy server (client).

    wire.py         serialize a PEFT adapter to chunks and reassemble on the far side
    publisher.py    AdapterPublisher -- the trainer hosts this; holds the latest adapter
    client.py       AdapterClient -- the policy server pulls the latest adapter on demand
    applier.py      AdapterApplier -- inject/swap a received adapter into a live policy
    service.proto   the AdapterService definition (regenerate stubs when it changes)

Regenerate the stubs from the policy_learning root:

    python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. \
        lora_finetuning/transport/service.proto
"""

from .applier import AdapterApplier
from .client import AdapterClient
from .publisher import AdapterPublisher
from .wire import AdapterVersion

__all__ = ["AdapterApplier", "AdapterClient", "AdapterPublisher", "AdapterVersion"]

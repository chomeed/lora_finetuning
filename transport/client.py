"""Policy-server-side gRPC client that pulls the latest LoRA adapter on demand.

Pull model: the caller decides *when* to fetch. ``fetch()`` sends one
``GetLatestAdapter`` request advertising the version already loaded; the trainer
replies with the newer adapter (streamed as chunks, reassembled into a local temp
dir) or an empty stream if there is nothing newer. It returns the received
``AdapterVersion`` or ``None``, and never raises -- a dropped connection just
yields ``None`` and the channel is rebuilt on the next call.

Because the server sends nothing when the client is already current, calling
``fetch()`` frequently (e.g. every action-chunk boundary) is cheap: it costs one
small round-trip and only transfers bytes when a new adapter actually exists.
"""

import logging

import grpc
from tqdm import tqdm

from . import service_pb2 as pb
from . import service_pb2_grpc as pb_grpc
from .wire import AdapterAssembler, AdapterVersion

logger = logging.getLogger("lora_adapter_client")

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
]


class AdapterClient:
    """Pulls adapters from a remote trainer, one request per ``fetch()`` call.

    ``root`` is where received adapters are materialized locally (defaults to a
    temp dir). Not thread-safe: call ``fetch()`` from a single thread.
    """

    def __init__(self, addr: str, root=None, keep_last: int = 3, progress: bool = False):
        self._addr = addr
        self._assembler = AdapterAssembler(root)
        self._keep_last = keep_last
        self._progress = progress  # show a tqdm bar for the chunk transfer
        self._loaded_version: int | None = None
        self._channel = None
        self._stub = None

    @property
    def loaded_version(self) -> int | None:
        return self._loaded_version

    def _stub_or_connect(self):
        if self._stub is None:
            self._channel = grpc.insecure_channel(self._addr, options=_GRPC_OPTIONS)
            self._stub = pb_grpc.AdapterServiceStub(self._channel)
        return self._stub

    def _reset_channel(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None

    def fetch(self) -> AdapterVersion | None:
        """Request the latest adapter now. Returns it if newer than loaded, else None."""
        have = self._loaded_version or 0
        bar = None
        try:
            stub = self._stub_or_connect()
            received = None
            for chunk in stub.GetLatestAdapter(pb.GetLatestRequest(have_version=have)):
                # total_bytes rides the first chunk; start the bar once we know the size.
                if self._progress and bar is None and chunk.meta.total_bytes:
                    bar = tqdm(
                        total=chunk.meta.total_bytes,
                        desc=f"pull adapter v{chunk.meta.version}",
                        unit="B",
                        unit_scale=True,
                        leave=False,
                    )
                if bar is not None:
                    bar.update(len(chunk.weights))
                meta = self._assembler.add(chunk)
                if meta is not None:
                    received = meta
            if received is not None:
                self._assembler.cleanup(self._keep_last)
                logger.info(
                    f"Fetched adapter v{received.version} (step {received.step}) -> {received.local_dir}"
                )
            return received
        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else "?"
            logger.warning(f"Adapter fetch failed ({code}); will retry on next fetch")
            self._reset_channel()
            return None
        finally:
            if bar is not None:
                bar.close()

    def mark_loaded(self, meta: AdapterVersion) -> None:
        self._loaded_version = meta.version

    def close(self) -> None:
        self._reset_channel()

"""Trainer-side gRPC server that hands the latest LoRA adapter to a policy server.

Pull model: the trainer keeps exactly one "latest" adapter in memory and answers
``GetLatestAdapter`` on demand. The client decides when to ask; the server just
returns the current adapter if it is newer than the client's version, else an
empty stream. Publishing is a cheap in-memory swap of the served adapter.

Usage (in trainer.py):

    publisher = AdapterPublisher(host="0.0.0.0", port=8090)
    publisher.start()
    ...
    publisher.publish(peft_model, version=v, step=step, loss=loss)
    ...
    publisher.stop()
"""

import logging
import os
import shutil
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc

from . import service_pb2_grpc as pb_grpc
from .wire import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHTS_FILE,
    AdapterVersion,
    iter_chunks,
    serialize_adapter,
)

logger = logging.getLogger("lora_adapter_publisher")

# We chunk ourselves, but bump the caps so a whole adapter comfortably fits.
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
]


class _AdapterServicer(pb_grpc.AdapterServiceServicer):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: AdapterVersion | None = None
        self._weights: bytes = b""

    def set_latest(self, meta: AdapterVersion, weights: bytes) -> None:
        with self._lock:
            self._latest = meta
            self._weights = weights

    def GetLatestAdapter(self, request, context):  # noqa: N802
        with self._lock:
            meta, weights = self._latest, self._weights

        if meta is None or meta.version <= request.have_version:
            return  # empty stream: nothing newer than the client already has

        for chunk in iter_chunks(meta, weights):
            if not context.is_active():
                return
            yield chunk
        logger.info(f"Served adapter v{meta.version} to {context.peer()}")


def _save_adapter(root: str | os.PathLike, version: int, config_json: str, weights: bytes) -> Path:
    """Write one adapter to ``<root>/v00000N/`` atomically (tmp dir + rename), so
    a crash mid-write can never leave a half-saved dir that resume would load."""
    root = Path(root)
    dest = root / f"v{version:06d}"
    tmp = root / f".v{version:06d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / ADAPTER_CONFIG_FILE).write_text(config_json or "{}")
    (tmp / ADAPTER_WEIGHTS_FILE).write_bytes(weights)
    if dest.exists():
        shutil.rmtree(dest)  # re-publish of the same version replaces it
    os.replace(tmp, dest)
    return dest


class AdapterPublisher:
    def __init__(self, host: str = "0.0.0.0", port: int = 8090, max_workers: int = 4):
        self._addr = f"{host}:{port}"
        self._servicer = _AdapterServicer()
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=_GRPC_OPTIONS,
        )
        pb_grpc.add_AdapterServiceServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(self._addr)
        self._started = False

    def start(self) -> None:
        self._server.start()
        self._started = True
        logger.info(f"AdapterPublisher serving on {self._addr}")

    def publish(
        self,
        peft_model,
        version: int,
        step: int,
        loss: float | None = None,
        save_dir: str | os.PathLike | None = None,
    ) -> AdapterVersion:
        """Serialize ``peft_model``'s adapter and make it the one the server serves.

        ``save_dir`` additionally writes a durable copy to ``<save_dir>/v00000N/``
        (the layout ``--resume_adapter_path`` loads); without it the adapter only
        exists in this process's memory. A failed save never blocks publishing."""
        config_json, weights = serialize_adapter(peft_model)
        meta = AdapterVersion(
            version=version,
            step=step,
            published_at=time.time(),
            loss=loss,
            config_json=config_json,
        )
        self._servicer.set_latest(meta, weights)
        logger.info(f"Published adapter v{version} (step {step}, {len(weights) / 1e6:.1f}MB)")
        if save_dir:
            try:
                dest = _save_adapter(save_dir, version, config_json, weights)
                logger.info(f"Saved adapter v{version} -> {dest}")
            except OSError as e:
                logger.warning(f"could not save adapter v{version} to {save_dir}: {e}")
        return meta

    def wait(self) -> None:
        """Block until the server terminates (e.g. keeps serving after training)."""
        self._server.wait_for_termination()

    def stop(self, grace: float = 2.0) -> None:
        if self._started:
            self._server.stop(grace)
            self._started = False

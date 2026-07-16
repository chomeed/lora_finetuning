"""Serialize a PEFT adapter to the wire and reassemble it on the far side.

Shared by the trainer (publisher) and the policy server (client). The design
keeps the receiving side simple: an adapter arrives as a stream of chunks, and
``AdapterAssembler`` lands it in a local temp dir as ``adapter_config.json`` +
``adapter_model.safetensors`` -- the exact layout PEFT's ``from_pretrained`` /
``load_peft_weights`` expect. The policy server's adapter-loading code is then
identical to the old filesystem path; only the *source* of the directory changed.
"""

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from peft.utils import get_peft_model_state_dict
from safetensors.torch import save as safetensors_save

from . import service_pb2 as pb

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"

# Chunk well under gRPC's 4MB default message cap; leaves headroom for framing.
CHUNK_BYTES = 1 << 20  # 1 MiB


@dataclass
class AdapterVersion:
    """Metadata for one published adapter (mirrors the ``AdapterMeta`` proto).

    ``local_dir`` is populated only on the receiving side, by ``AdapterAssembler``,
    once the adapter has been fully materialized to disk.
    """

    version: int
    step: int
    published_at: float
    loss: float | None = None
    config_json: str | None = None
    local_dir: str | None = None


def serialize_adapter(peft_model, adapter_name: str = "default") -> tuple[str, bytes]:
    """Return ``(config_json, weights_bytes)`` for ``peft_model``'s adapter, in memory.

    Only the adapter tensors are serialized (a few MB), never the base policy --
    the same contract the old ``publish_adapter`` had, without touching disk.
    """
    state_dict = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
    # safetensors requires contiguous CPU tensors.
    state_dict = {k: v.detach().to("cpu").contiguous() for k, v in state_dict.items()}
    weights = safetensors_save(state_dict)

    config = peft_model.peft_config[adapter_name].to_dict()
    config_json = json.dumps(_json_safe(config), indent=2)
    return config_json, weights


def _json_safe(obj):
    """PEFT configs can hold sets/enums that ``json`` refuses; coerce them."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    if hasattr(obj, "value"):  # enum-like
        return obj.value
    return obj


def iter_chunks(meta: AdapterVersion, weights: bytes):
    """Yield ``AdapterChunk`` messages for one adapter. ``meta`` rides the first chunk."""
    total = len(weights)
    proto_meta = pb.AdapterMeta(
        version=meta.version,
        step=meta.step,
        loss=meta.loss if meta.loss is not None else 0.0,
        has_loss=meta.loss is not None,
        published_at=meta.published_at,
        config_json=meta.config_json or "",
        total_bytes=total,
    )

    if total == 0:
        yield pb.AdapterChunk(meta=proto_meta, weights=b"", last=True)
        return

    first = True
    for start in range(0, total, CHUNK_BYTES):
        end = min(start + CHUNK_BYTES, total)
        yield pb.AdapterChunk(
            meta=proto_meta if first else pb.AdapterMeta(),
            weights=weights[start:end],
            last=end >= total,
        )
        first = False


class AdapterAssembler:
    """Reassembles a chunk stream into a local adapter dir.

    ``add(chunk)`` buffers weights and returns an ``AdapterVersion`` (with
    ``local_dir`` set) when a ``last`` chunk completes an adapter, else ``None``.
    Each completed adapter gets its own temp dir; ``cleanup`` prunes old ones.
    """

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root) if root else Path(tempfile.gettempdir()) / "lora_adapters_rx"
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta: AdapterVersion | None = None
        self._buf = bytearray()
        self._dirs: list[Path] = []

    def add(self, chunk) -> AdapterVersion | None:
        if chunk.HasField("meta") and chunk.meta.config_json:
            # First chunk of a new adapter.
            m = chunk.meta
            self._meta = AdapterVersion(
                version=m.version,
                step=m.step,
                published_at=m.published_at,
                loss=m.loss if m.has_loss else None,
                config_json=m.config_json,
            )
            self._buf = bytearray()

        self._buf.extend(chunk.weights)

        if not chunk.last:
            return None
        if self._meta is None:
            # A trailing chunk with no preceding meta: drop it defensively.
            self._buf = bytearray()
            return None

        meta = self._materialize(self._meta, bytes(self._buf))
        self._meta = None
        self._buf = bytearray()
        return meta

    def _materialize(self, meta: AdapterVersion, weights: bytes) -> AdapterVersion:
        dest = self._root / f"v{meta.version:06d}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ADAPTER_CONFIG_FILE).write_text(meta.config_json or "{}")
        (dest / ADAPTER_WEIGHTS_FILE).write_bytes(weights)
        self._dirs.append(dest)
        return replace(meta, local_dir=str(dest))

    def cleanup(self, keep_last: int = 3) -> None:
        import shutil

        while len(self._dirs) > keep_last:
            shutil.rmtree(self._dirs.pop(0), ignore_errors=True)

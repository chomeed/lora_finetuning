"""Filesystem hand-off protocol for LoRA adapters: learner -> policy server.

Layout under ``adapter_dir``:

    adapter_dir/
      latest.json          {"version": 42, "step": 21000, "dirname": "v000042", ...}
      v000042/
        adapter_config.json
        adapter_model.safetensors
      v000041/
        ...

The learner saves a new adapter into a temporary directory, renames it into
place, and only then rewrites ``latest.json`` (also via rename). Both renames
are atomic on POSIX within one filesystem, so a reader that keys off
``latest.json`` never observes a half-written adapter. That is the whole
concurrency story here -- there are no locks between the two processes.
"""

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LATEST_FILE = "latest.json"
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"

_VERSION_DIR_RE = re.compile(r"^v(\d{6})$")


@dataclass(frozen=True)
class AdapterVersion:
    """Metadata for one published adapter, mirrored into ``latest.json``."""

    version: int
    step: int
    dirname: str
    published_at: float
    loss: float | None = None

    def path(self, root: str | Path) -> Path:
        return Path(root) / self.dirname


def _version_dirs(root: Path) -> list[tuple[int, Path]]:
    if not root.is_dir():
        return []
    out = []
    for child in root.iterdir():
        m = _VERSION_DIR_RE.match(child.name)
        if m and child.is_dir():
            out.append((int(m.group(1)), child))
    return sorted(out, key=lambda kv: kv[0])


def next_version(root: str | Path) -> int:
    """First unused version number, so a restarted learner never overwrites history."""
    existing = _version_dirs(Path(root))
    return existing[-1][0] + 1 if existing else 1


def read_latest(root: str | Path) -> AdapterVersion | None:
    """Return the currently published version, or None if nothing is published yet.

    Returns None rather than raising on a malformed/torn read so callers can
    simply retry on the next poll.
    """
    latest = Path(root) / LATEST_FILE
    try:
        payload = json.loads(latest.read_text())
        return AdapterVersion(**payload)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _write_latest(root: Path, meta: AdapterVersion) -> None:
    tmp = root / f".{LATEST_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(meta), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, root / LATEST_FILE)


def _prune(root: Path, keep_last: int, protect: int) -> None:
    if keep_last <= 0:
        return
    versions = _version_dirs(root)
    for version, path in versions[:-keep_last]:
        if version == protect:
            continue
        shutil.rmtree(path, ignore_errors=True)


def publish_adapter(
    peft_model,
    root: str | Path,
    version: int,
    step: int,
    loss: float | None = None,
    keep_last: int = 5,
) -> AdapterVersion:
    """Write ``peft_model``'s adapter as version ``version`` and make it current.

    Only the adapter weights and config are written (a few MB), never the base
    policy. Safe to call while a policy server is polling the same directory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    dirname = f"v{version:06d}"
    dest = root / dirname
    tmp = root / f".tmp.{dirname}.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)

    # PEFT writes the "default" adapter at the root of the save directory.
    peft_model.save_pretrained(str(tmp))

    for required in (ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE):
        if not (tmp / required).is_file():
            shutil.rmtree(tmp, ignore_errors=True)
            raise FileNotFoundError(
                f"PEFT did not write {required} into {tmp}. Contents: "
                f"{[p.name for p in tmp.iterdir()] if tmp.is_dir() else '<missing>'}"
            )

    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)

    meta = AdapterVersion(
        version=version,
        step=step,
        dirname=dirname,
        published_at=time.time(),
        loss=loss,
    )
    _write_latest(root, meta)
    _prune(root, keep_last=keep_last, protect=version)
    return meta


class AdapterWatcher:
    """Polls an adapter directory for versions newer than the one already loaded.

    ``poll()`` is cheap enough to call on every inference: the common case is a
    single ``stat`` on ``latest.json``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._latest_file = self.root / LATEST_FILE
        self._loaded_version: int | None = None
        self._seen_mtime: int | None = None

    @property
    def loaded_version(self) -> int | None:
        return self._loaded_version

    def poll(self) -> AdapterVersion | None:
        try:
            mtime = self._latest_file.stat().st_mtime_ns
        except OSError:
            return None

        if mtime == self._seen_mtime:
            return None

        meta = read_latest(self.root)
        if meta is None:
            return None  # torn read; retry next poll without caching mtime

        adapter_dir = meta.path(self.root)
        if not (adapter_dir / ADAPTER_WEIGHTS_FILE).is_file():
            return None  # published dir vanished (pruned?); retry next poll

        self._seen_mtime = mtime
        if self._loaded_version is not None and meta.version <= self._loaded_version:
            return None
        return meta

    def mark_loaded(self, meta: AdapterVersion) -> None:
        self._loaded_version = meta.version

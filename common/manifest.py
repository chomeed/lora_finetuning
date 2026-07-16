"""Dagger manifest for a growing LeRobot dataset.

Records, per converted episode, *which policy generated the rollout* and how much
of it was human intervention. This is the provenance the LeRobot dataset itself
doesn't carry: the dataset stores the per-frame ``intervention`` flag, but not the
identity of the policy the human was correcting.

Stored as append-only JSONL at ``<dataset_root>/meta/dagger_manifest.jsonl`` --
one row per episode, in conversion order. Append-only means a crash mid-run never
corrupts earlier rows, and the file lines up with the dataset's own episode
indexing (row i describes ``episode_index`` i, barring the harmless duplicate a
crash between finalize and manifest-append could leave).

    from lora_finetuning.common.manifest import DaggerManifest, ManifestEntry
    m = DaggerManifest(dataset_root)
    m.append(ManifestEntry(episode_index=12, source_file="episode_012.h5",
                            task="board_insertion", policy_id="run3/step2000",
                            n_frames=1544, n_intervention_frames=88))
    print(m.summary())   # {'run3/step2000': {'episodes': 1, 'frames': 1544, ...}}
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = "dagger_manifest.jsonl"


@dataclass
class ManifestEntry:
    """One converted episode's provenance."""

    episode_index: int
    source_file: str
    task: str
    policy_id: str  # which policy generated the rollout ("unknown" if untagged)
    n_frames: int
    n_intervention_frames: int
    converted_at: float | None = None  # unix epoch seconds; stamped by the caller
    # Adapter version + trainer step the LoRAPolicyServer was serving at
    # conversion time (from its status file), if available.
    adapter_version: int | None = None
    trainer_step: int | None = None
    extra: dict = field(default_factory=dict)  # room for future fields

    @property
    def intervention_fraction(self) -> float:
        return self.n_intervention_frames / self.n_frames if self.n_frames else 0.0

    def to_json(self) -> str:
        d = asdict(self)
        d["intervention_fraction"] = round(self.intervention_fraction, 6)
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        d = dict(d)
        d.pop("intervention_fraction", None)  # derived; not a constructor arg
        known = {f for f in cls.__dataclass_fields__}  # noqa: E501
        extra = d.pop("extra", {}) or {}
        extra.update({k: d.pop(k) for k in list(d) if k not in known})
        return cls(extra=extra, **d)


POLICY_STATUS_NAME = "policy_version.json"


def write_policy_status(path: str | os.PathLike, version: int, trainer_step=None, loss=None) -> None:
    """Atomically write the currently-serving adapter version to ``path`` (JSON).

    Called by the LoRAPolicyServer on every adapter swap so a co-located
    converter can read which policy version is live. Atomic (temp + rename) so a
    concurrent reader never sees a half-written file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": int(version), "trainer_step": trainer_step, "loss": loss}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, p)


def read_policy_status(path: str | os.PathLike) -> dict | None:
    """Read a policy version status file, or None if absent/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


class DaggerManifest:
    """Append-only reader/writer for one dataset's dagger manifest."""

    def __init__(self, dataset_root: str | os.PathLike):
        self.path = Path(dataset_root) / "meta" / MANIFEST_NAME

    def append(self, entry: ManifestEntry) -> None:
        """Durably append one row (parent dir created on demand, fsync'd)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(entry.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[ManifestEntry]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(ManifestEntry.from_dict(json.loads(line)))
        return rows

    def summary(self) -> dict[str, dict]:
        """Per-policy totals: episodes, frames, intervention frames + fraction."""
        agg: dict[str, dict] = {}
        for e in self.read_all():
            s = agg.setdefault(
                e.policy_id, {"episodes": 0, "frames": 0, "intervention_frames": 0}
            )
            s["episodes"] += 1
            s["frames"] += e.n_frames
            s["intervention_frames"] += e.n_intervention_frames
        for s in agg.values():
            s["intervention_fraction"] = (
                round(s["intervention_frames"] / s["frames"], 6) if s["frames"] else 0.0
            )
        return agg

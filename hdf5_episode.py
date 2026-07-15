"""Read one recorded HDF5 episode into arrays a LeRobotDataset can ingest.

Self-contained copy of the read/decode half of the Orin-side offline converter
(``orin_demo_collection/convert_to_lerobot.py``) so the workstation converter has
no dependency on the ROS package or its Python env. Keep the two in sync if the
recording schema changes.

Schema (attrs on the HDF5 root):
  schema_version    "1.0" (per-group datasets) or "2.0" (flat state/action arrays)
  fps               int
  task              str
  {state,action}_field_order / _field_dims   concatenation order + per-field widths
  <field>_joint_names                        per-column names, when width matches

Datasets:
  /observation/state   [N, state_dim] float
  /action              [N, act_dim]   float
  /timestamp           [N]            int64 ns
  /observation/images/{head,left_wrist,right_wrist}   [N] object (JPEG bytes)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# Concatenation order for schema 1.0 (per-group datasets). Schema 2.0 carries its
# own *_field_order attrs and these are unused.
OBS_FIELD_ORDER = ["arm_left", "arm_right", "head", "lift", "gripper_left", "gripper_right"]
ACT_FIELD_ORDER = ["arm_left", "arm_right", "head", "lift", "gripper_left", "gripper_right"]

SUPPORTED_SCHEMA_VERSIONS = ("1.0", "2.0")

IMAGE_NAMES = ("head", "left_wrist", "right_wrist")


class SchemaVersionMismatch(RuntimeError):
    pass


def _as_str(s) -> str:
    return s.decode() if isinstance(s, (bytes, bytearray)) else str(s)


def _column_names(f: h5py.File, field_order, field_dims) -> list[str]:
    """Per-column names for a flat state/action vector. Uses the
    ``<field>_joint_names`` HDF5 attr when its length matches the field's width;
    otherwise falls back to ``<field>`` (width 1) or ``<field>_<i>`` (e.g. the
    12-DoF gripper state, which has no joint-name attr)."""
    names: list[str] = []
    for fname, dim in zip(
        (_as_str(s) for s in field_order),
        (int(d) for d in field_dims),
    ):
        attr = f.attrs.get(f"{fname}_joint_names")
        if attr is not None and len(attr) == dim:
            names.extend(_as_str(s) for s in attr)
        elif dim == 1:
            names.append(fname)
        else:
            names.extend(f"{fname}_{i}" for i in range(dim))
    return names


def read_episode_arrays(path: Path) -> dict:
    """Read one HDF5 episode. Returns a dict with keys:

        fps, task, observation.state [N, D], action [N, A], timestamp_ns [N],
        state_names [D], action_names [A],
        observation.images.{head,left_wrist,right_wrist}  list[bytes] (length N)

    Raises SchemaVersionMismatch if the file's ``schema_version`` is unsupported.
    """
    with h5py.File(path, "r") as f:
        ver = str(f.attrs.get("schema_version", ""))
        if ver not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionMismatch(
                f"{path}: schema_version {ver!r} not in supported set {SUPPORTED_SCHEMA_VERSIONS!r}"
            )

        fps = int(f.attrs.get("fps", 30))
        task = str(f.attrs.get("task", ""))

        if ver == "2.0":
            # Schema 2.0: state and action are already flat arrays.
            observation_state = f["/observation/state"][:].astype(np.float32, copy=False)
            action = f["/action"][:].astype(np.float32, copy=False)
            state_names = _column_names(f, f.attrs["state_field_order"], f.attrs["state_field_dims"])
            action_names = _column_names(f, f.attrs["action_field_order"], f.attrs["action_field_dims"])
        else:
            # Schema 1.0: per-group datasets concatenated in OBS/ACT_FIELD_ORDER.
            state_parts = []
            for k in OBS_FIELD_ORDER:
                d = f[f"/observation/state/{k}"][:]
                if d.ndim == 1:
                    d = d[:, None]
                state_parts.append(d.astype(np.float32, copy=False))
            observation_state = np.concatenate(state_parts, axis=1)

            action_parts = []
            for k in ACT_FIELD_ORDER:
                d = f[f"/action/{k}"][:]
                if d.ndim == 1:
                    d = d[:, None]
                action_parts.append(d.astype(np.float32, copy=False))
            action = np.concatenate(action_parts, axis=1)

            state_names = _column_names(f, OBS_FIELD_ORDER, [p.shape[1] for p in state_parts])
            action_names = _column_names(f, ACT_FIELD_ORDER, [p.shape[1] for p in action_parts])

        timestamp_ns = f["/timestamp"][:]

        images = {}
        for name in IMAGE_NAMES:
            ds = f[f"/observation/images/{name}"]
            images[f"observation.images.{name}"] = [bytes(ds[i]) for i in range(ds.shape[0])]

    return {
        "fps": fps,
        "task": task,
        "observation.state": observation_state,
        "action": action,
        "timestamp_ns": timestamp_ns,
        "state_names": state_names,
        "action_names": action_names,
        **images,
    }


def decode_jpeg(buf: bytes) -> np.ndarray:
    """Decode a JPEG byte string to an H x W x 3 RGB uint8 array."""
    import cv2  # lazy: only the conversion path needs it

    arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None -- corrupt JPEG?")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_frame_images(ep: dict, idx: int) -> dict:
    """Decode all camera JPEGs for one frame; runs in a worker thread."""
    return {
        f"observation.images.{name}": decode_jpeg(ep[f"observation.images.{name}"][idx])
        for name in IMAGE_NAMES
    }

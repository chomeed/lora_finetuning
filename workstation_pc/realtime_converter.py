"""Realtime HDF5 -> LeRobot converter.

The workstation (rllab4) leg of the data loop: the robot uploads finished
``episode_*.h5`` files into an ingest directory (atomic rename from ``.partial``,
so a visible ``.h5`` is always complete); this daemon watches that directory and
appends each new episode to a single growing LeRobot dataset -- the same dataset
``trainer.py`` trains its LoRA adapter on. Point the trainer at the same
``--dataset_root``/``--dataset_repo_id`` and it picks up new episodes on its next
(re)start.

Two properties make it safe to run next to a live, high-priority policy server:

  * **It yields.** The process is CPU-niced and (best-effort) IO-idle, and it
    *pauses* whole-episode conversion whenever the GPU is busy (utilization above
    a threshold) or a pause-file is present. JPEG decode + AV1 encode are the
    expensive parts and both happen while paused-checks gate them.
  * **It is crash-safe and idempotent.** The ingest dir *is* the queue: a file
    sitting in it is pending. Each episode is converted, then ``finalize()``d
    into a valid, readable checkpoint, and only then is the source ``.h5``
    deleted (the robot keeps the raw recording). A crash mid-episode leaves the
    source in place, so it is simply reconverted next run -- no ledger to keep in
    sync or lose. Files that fail or can't be converted are moved to a ``failed/``
    subdir so they aren't retried on every scan. (A crash in the sub-millisecond
    window between finalize and delete re-appends one duplicate episode next run
    -- harmless, just a repeated demo.)

Shortcut (installed by `pip install -e .`): ``ws-real-time-converter``. Both
hyphen and underscore flag spellings are accepted (``--num-demos`` == ``--num_demos``).

    # SIRIUS round mode: project to the policy's schema, and every 30 episodes
    # roll over to the next <task>_sirius_round<N> dataset -- continuously.
    ws-real-time-converter \
        --ingest_dir=/data/incoming/tmp_demo \
        --dagger_datasets_dir=/data/lerobot/dagger \
        --default_task=board_insertion \
        --mode=insertion_15 \
        --num-demos 30 \
        --policy_status_file=/data/lerobot/policy_version.json

    # single-dataset mode: append everything into one dataset
    ws-real-time-converter \
        --ingest_dir=/data/incoming/tmp_demo \
        --dataset_repo_id=chomeed/board_insertion_dagger \
        --dataset_root=/data/lerobot/board_insertion_dagger \
        --default_task=board_insertion --mode=insertion_15

    # convert whatever is already there and exit (catch-up / testing):
    ws-real-time-converter ... --run_once=true

Equivalent module form (from ``policy_learning/``, in the ``policy_learning``
conda env): ``python -m lora_finetuning.workstation_pc.realtime_converter``.

``--mode`` projects the full-rig recording down to a policy's I/O schema
(common/schema.py); every converted episode carries the per-frame
``intervention`` flag and gets a row in the round dataset's
``meta/dagger_manifest.jsonl`` naming the policy version that generated it.
"""

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np

from ..common._cli import normalize_hyphen_flags
from ..common.configs import RealtimeConverterConfig
from ..common.hdf5_episode import (
    SchemaVersionMismatch,
    decode_frame_images,
    decode_jpeg,
    read_episode_arrays,
)
from ..common.manifest import DaggerManifest, ManifestEntry, read_policy_status
from ..common.schema import ACTION_KEYS, STATE_KEYS, get_schema, projection_indices

logger = logging.getLogger("realtime_converter")


# ── process priority ──────────────────────────────────────────────────────
def apply_process_priority(nice: int, ionice_idle: bool) -> None:
    """Nice this process (CPU) and, best-effort, make its disk IO idle-class so
    it yields to the policy server. Child encode processes inherit both."""
    try:
        os.nice(nice)  # relative increment; positive = lower priority
        logger.info(f"CPU niceness raised by {nice} (now {os.nice(0)})")
    except OSError as e:
        logger.warning(f"could not renice: {e}")
    if ionice_idle:
        try:
            subprocess.run(
                ["ionice", "-c3", "-p", str(os.getpid())],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("IO class set to idle (ionice -c3)")
        except (OSError, subprocess.CalledProcessError) as e:
            logger.warning(f"could not ionice: {e}")


# ── throttle ──────────────────────────────────────────────────────────────
class ThrottleController:
    """Blocks conversion while the GPU is busy or a pause-file is present.

    GPU gating has hysteresis: it engages at ``gpu_util_pause`` and only releases
    at/below ``gpu_util_resume``, so a policy server hovering near the threshold
    doesn't cause rapid start/stop churn. If ``nvidia-smi`` can't be read the GPU
    signal is treated as "unknown" and ignored (the pause-file still works)."""

    def __init__(self, cfg: RealtimeConverterConfig, should_stop):
        self.cfg = cfg
        self._should_stop = should_stop
        self.pause_file = Path(cfg.pause_file) if cfg.pause_file else Path(cfg.ingest_dir) / ".pause"
        self._gpu_paused = False  # current hysteresis state
        self._nvidia_smi_ok = True

    def _gpu_util(self) -> int | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.cfg.gpu_index}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            return int(out.splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
            if self._nvidia_smi_ok:  # log the first failure only
                logger.warning(f"nvidia-smi unreadable, ignoring GPU throttle signal: {e}")
                self._nvidia_smi_ok = False
            return None

    def _blocked_reason(self) -> str | None:
        """Return a human string if conversion should pause right now, else None."""
        if self.pause_file.exists():
            return f"pause-file {self.pause_file} present"
        util = self._gpu_util()
        if util is None:
            self._gpu_paused = False
            return None
        # hysteresis
        if self._gpu_paused:
            self._gpu_paused = util > self.cfg.gpu_util_resume
        else:
            self._gpu_paused = util >= self.cfg.gpu_util_pause
        if self._gpu_paused:
            return f"GPU {self.cfg.gpu_index} at {util}% (>= {self.cfg.gpu_util_pause}%)"
        return None

    def wait_until_clear(self) -> None:
        """Block until conversion is allowed (or a stop is requested)."""
        announced = None
        while not self._should_stop():
            reason = self._blocked_reason()
            if reason is None:
                if announced is not None:
                    logger.info("throttle cleared, resuming conversion")
                return
            if reason != announced:
                logger.info(f"throttled: {reason}")
                announced = reason
            time.sleep(self.cfg.throttle_poll_s)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── converter ─────────────────────────────────────────────────────────────
class RealtimeConverter:
    def __init__(self, cfg: RealtimeConverterConfig):
        self.cfg = cfg
        self.ingest_dir = Path(cfg.ingest_dir)
        # Where handled files go. Defaults are subdirs of the ingest dir, so the
        # top-level (non-recursive) glob never re-scans them.
        self.converted_dir = Path(cfg.converted_dir) if cfg.converted_dir else self.ingest_dir / "converted"
        self.failed_dir = Path(cfg.failed_dir) if cfg.failed_dir else self.ingest_dir / "failed"
        self._stop = False
        self.throttle = ThrottleController(cfg, lambda: self._stop)
        self.default_policy_id = cfg.dagger_policy or "unknown"
        # Observation/action schema: project the full-rig recording down to the
        # chosen mode's channels + cameras. For "full" these indices are the
        # identity (all 41 state / 19 action columns).
        self.schema = get_schema(cfg.mode)
        self.state_idx = projection_indices(STATE_KEYS, self.schema.state_keys)
        self.action_idx = projection_indices(ACTION_KEYS, self.schema.action_keys)

        # ── dataset target ──────────────────────────────────────────────
        # ROUND mode (dagger_datasets_dir set): write <task>_sirius_round<N>
        # datasets, rolling to the next once one holds num_demos episodes.
        # SINGLE mode: one fixed dataset (dataset_repo_id/dataset_root).
        self.round_mode = cfg.dagger_datasets_dir is not None
        if self.round_mode:
            self.rounds_dir = Path(cfg.dagger_datasets_dir)
            # Resume the highest not-yet-full round, else start the next one.
            self.round_idx = self._initial_round()
        else:
            self.rounds_dir = None
            self.round_idx = None
        self._set_round_target()

    # -- round targeting --
    def _round_dirname(self, n: int) -> str:
        return f"{self.cfg.default_task}_sirius_round{n}"

    def _round_root(self, n: int) -> Path:
        return self.rounds_dir / self._round_dirname(n)

    def _episode_count_at(self, root: Path) -> int:
        info = root / "meta" / "info.json"
        if not info.exists():
            return 0
        try:
            return int(json.loads(info.read_text()).get("total_episodes", 0))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            return 0

    def _initial_round(self) -> int:
        """Lowest round >= 1 that isn't full yet (resumes a partial round on
        restart; skips full ones)."""
        n = 1
        while self._episode_count_at(self._round_root(n)) >= self.cfg.num_demos:
            n += 1
        return n

    def _set_round_target(self) -> None:
        """Point self.root/repo_id/manifest at the current target dataset."""
        if self.round_mode:
            self.root = self._round_root(self.round_idx)
            self.repo_id = f"{self.cfg.repo_namespace}/{self._round_dirname(self.round_idx)}"
        else:
            self.root = Path(self.cfg.dataset_root)
            self.repo_id = self.cfg.dataset_repo_id
        # Each round dataset carries its own dagger manifest under meta/.
        self.manifest = DaggerManifest(self.root)

    def _advance_round(self) -> None:
        self.round_idx += 1
        self._set_round_target()
        logger.info(f"rolling to round {self.round_idx}: {self.repo_id} at {self.root}")

    # -- lifecycle --
    def request_stop(self, *_):
        if not self._stop:
            logger.info("stop requested; will exit after the current episode")
        self._stop = True

    def _dataset_exists(self) -> bool:
        return (self.root / "meta" / "info.json").exists()

    def _dataset_episode_count(self) -> int:
        """Total episodes in the current target dataset (0 if not created)."""
        return self._episode_count_at(self.root)

    @staticmethod
    def _writer_episode_count(ds) -> int:
        """Total episodes known to an open LeRobotDataset (post-finalize)."""
        n = getattr(ds, "num_episodes", None)
        if isinstance(n, int):
            return n
        meta = getattr(ds, "meta", None)
        n = getattr(meta, "total_episodes", None)
        if isinstance(n, int):
            return n
        return len(getattr(meta, "episodes", []) or [])

    # -- scanning --
    def _ready_files(self) -> list[Path]:
        """Complete, pending episode files, oldest first. A file is pending simply
        by being in the ingest dir: handled files are deleted or moved aside."""
        now = time.time()
        out = []
        for p in sorted(self.ingest_dir.glob(self.cfg.glob)):
            if p.suffix != ".h5" or p.name.endswith(".partial"):
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue  # vanished between glob and stat
            if now - st.st_mtime < self.cfg.min_age_s:
                continue  # too fresh; let any in-flight write settle
            out.append(p)
        return out

    # -- disposition (the queue) --
    def _with_sidecar(self, path: Path):
        """The h5 and its optional .sha256 sidecar, whichever exist."""
        for p in (path, path.with_name(path.name + ".sha256")):
            if p.exists():
                yield p

    def _delete_source(self, path: Path) -> None:
        for p in self._with_sidecar(path):
            p.unlink()

    def _move_source(self, path: Path, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for p in self._with_sidecar(path):
            shutil.move(str(p), str(dest_dir / p.name))

    # -- conversion --
    def _verify(self, path: Path) -> None:
        """Optional checksum verification against a <name>.sha256 sidecar."""
        if not self.cfg.verify_checksum:
            return
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.exists():
            return
        expected = sidecar.read_text().split()[0].strip()
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch (sidecar {expected[:12]}.., file {actual[:12]}..)")

    def _open_writer(self, ep: dict):
        """Create the dataset on the first episode, else resume for appending."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Streaming knobs are shared by create() and resume(); with streaming on,
        # frames are encoded as they're added (no temp PNGs, encode overlaps decode).
        stream_kwargs = dict(
            streaming_encoding=self.cfg.streaming_encoding,
            vcodec=self.cfg.vcodec,
            encoder_queue_maxsize=self.cfg.encoder_queue_maxsize,
            encoder_threads=self.cfg.encoder_threads,
        )

        if self._dataset_exists():
            return LeRobotDataset.resume(self.repo_id, root=str(self.root), **stream_kwargs)

        image_shapes = {}
        for name in self.schema.image_names:
            h, w, c = decode_jpeg(ep[f"observation.images.{name}"][0]).shape
            image_shapes[name] = (c, h, w)  # CHW to match ('channels','height','width')
        # Feature names come from the schema (projected channels), so the
        # on-disk dataset matches the policy's I/O layout exactly.
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.schema.state_keys),),
                "names": list(self.schema.state_keys),
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.schema.action_keys),),
                "names": list(self.schema.action_keys),
            },
            # Per-frame dagger flag: 1 where a human took over from the policy.
            # Always present on datasets this converter creates, so downstream
            # (e.g. the trainer's episode/frame filtering) can rely on it.
            "intervention": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
        }
        for name in self.schema.image_names:
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": image_shapes[name],
                "names": ("channels", "height", "width"),
            }
        logger.info(f"creating dataset {self.repo_id} at {self.root}")
        return LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=ep["fps"] or self.cfg.fps,
            root=str(self.root),
            features=features,
            **stream_kwargs,
        )

    def _convert_one(self, path: Path) -> int:
        """Read, append, and finalize one episode into the growing dataset.

        Returns the number of frames written; ``0`` means the episode was
        skipped (empty or filtered out). Raises on any failure (the caller
        quarantines the file and moves on)."""
        self._verify(path)
        ep = read_episode_arrays(path)

        task = ep["task"] or self.cfg.default_task
        if self.cfg.task_filter and task != self.cfg.task_filter:
            logger.info(f"skipping {path.name}: task {task!r} != filter {self.cfg.task_filter!r}")
            return 0
        n_rows = ep["observation.state"].shape[0]
        if n_rows == 0:
            logger.info(f"skipping {path.name}: 0 frames")
            return 0

        # Projection expects a full-rig recording; a differently-shaped episode
        # would silently misalign channels, so refuse it (caller quarantines).
        s_dim, a_dim = ep["observation.state"].shape[1], ep["action"].shape[1]
        if s_dim != len(STATE_KEYS) or a_dim != len(ACTION_KEYS):
            raise ValueError(
                f"expected full-rig {len(STATE_KEYS)}-D state / {len(ACTION_KEYS)}-D action, "
                f"got {s_dim}/{a_dim} -- mode projection needs the full recording"
            )
        # A camera can be present in the schema but empty on disk (e.g. the
        # 'third' cam is declared yet 0-byte on rigs that didn't run it) --
        # treat those as missing so the failure is a clear quarantine reason
        # rather than a cryptic cv2.imdecode error mid-episode.
        missing = [
            n
            for n in self.schema.image_names
            if f"observation.images.{n}" not in ep or len(ep[f"observation.images.{n}"][0]) == 0
        ]
        if missing:
            raise KeyError(
                f"mode {self.cfg.mode!r} needs camera(s) {missing} but the episode is "
                f"missing or empty there (available: {ep['available_images']})"
            )
        # Project state/action down to the mode's channels ("full" = identity).
        ep["observation.state"] = ep["observation.state"][:, self.state_idx]
        ep["action"] = ep["action"][:, self.action_idx]

        ds = self._open_writer(ep)
        # Old datasets created before the intervention feature existed won't have
        # the column; only emit it when the (resumed) schema includes it so
        # appends stay schema-compatible. Freshly created datasets always do.
        has_intervention = "intervention" in getattr(ds.meta, "features", {})
        try:
            # Pre-decode JPEGs in worker threads while add_frame consumes; cv2
            # releases the GIL so decode scales across cores.
            with ThreadPoolExecutor(max_workers=self.cfg.num_decode_workers) as ex:
                in_flight: deque[tuple[int, Future]] = deque()
                prefetch = self.cfg.num_decode_workers + 2
                next_submit = 0
                while in_flight or next_submit < n_rows:
                    while next_submit < n_rows and len(in_flight) < prefetch:
                        in_flight.append(
                            (next_submit, ex.submit(decode_frame_images, ep, next_submit, self.schema.image_names))
                        )
                        next_submit += 1
                    idx, fut = in_flight.popleft()
                    frame = {
                        "observation.state": ep["observation.state"][idx],
                        "action": ep["action"][idx],
                        "task": task,
                        **fut.result(),
                    }
                    if has_intervention:
                        frame["intervention"] = np.asarray([ep["intervention"][idx]], dtype=np.int64)
                    ds.add_frame(frame)
            ds.save_episode()
            # finalize() flushes parquet footers + info.json: the episode is now a
            # valid, readable checkpoint. Only after this do we delete the source.
            ds.finalize()
        except BaseException:
            # Drop the half-built episode buffer so a retry starts clean.
            try:
                if ds.has_pending_frames():
                    ds.clear_episode_buffer()
            except Exception:
                pass
            raise

        # Record dagger provenance for the episode we just finalized. The
        # policy VERSION comes from the episode's own tag if the recorder set
        # one, else from the LoRAPolicyServer's live status file (--policy_status_file).
        adapter_version = ep.get("policy_version")
        trainer_step = None
        if adapter_version is None and self.cfg.policy_status_file:
            status = read_policy_status(self.cfg.policy_status_file)
            if status is not None:
                adapter_version = status.get("version")
                trainer_step = status.get("trainer_step")
        self.manifest.append(
            ManifestEntry(
                episode_index=self._writer_episode_count(ds) - 1,
                source_file=path.name,
                task=task,
                policy_id=ep.get("policy_id") or self.default_policy_id,
                n_frames=n_rows,
                n_intervention_frames=int(np.asarray(ep["intervention"]).sum()),
                converted_at=time.time(),
                adapter_version=adapter_version,
                trainer_step=trainer_step,
            )
        )
        return n_rows

    # -- main loop --
    def run(self) -> int:
        apply_process_priority(self.cfg.nice, self.cfg.ionice_idle)
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        mode_desc = "run-once" if self.cfg.run_once else "daemon"
        if self.round_mode:
            logger.info(
                f"watching {self.ingest_dir}/{self.cfg.glob} -> round datasets under "
                f"{self.rounds_dir} ({self.cfg.num_demos} episodes/round, {mode_desc}); "
                f"resuming at round {self.round_idx}: {self.repo_id}"
            )
        else:
            logger.info(
                f"watching {self.ingest_dir}/{self.cfg.glob} -> {self.repo_id} ({mode_desc})"
            )
            if self.cfg.num_demos is not None:
                have = self._dataset_episode_count()
                if have >= self.cfg.num_demos:
                    logger.info(
                        f"dataset already holds {have} episode(s) >= num_demos={self.cfg.num_demos}; "
                        "nothing to convert"
                    )
                    return 0
                logger.info(
                    f"will stop once the dataset reaches num_demos={self.cfg.num_demos} "
                    f"episode(s) (currently {have})"
                )

        n_ok = 0
        while not self._stop:
            batch = self._ready_files()
            if not batch:
                if self.cfg.run_once:
                    break
                time.sleep(self.cfg.poll_interval_s)
                continue

            for path in batch:
                if self._stop:
                    break
                self.throttle.wait_until_clear()
                if self._stop:
                    break
                try:
                    t0 = time.perf_counter()
                    n = self._convert_one(path)
                except (SchemaVersionMismatch, RuntimeError, OSError, KeyError, ValueError) as e:
                    logger.error(f"FAILED {path.name}: {e} -> {self.failed_dir}")
                    self._move_source(path, self.failed_dir)
                    continue
                if n == 0:  # filtered / empty -> quarantine so we don't rescan it forever
                    self._move_source(path, self.failed_dir)
                    continue
                # Success: the episode is a durable checkpoint; drop the raw source.
                # (The robot keeps its own copy of the recording.)
                if self.cfg.delete_on_success:
                    self._delete_source(path)
                    disp = "deleted"
                else:
                    self._move_source(path, self.converted_dir)
                    disp = f"moved -> {self.converted_dir}"
                n_ok += 1
                logger.info(
                    f"converted {path.name}: {n} frames in {time.perf_counter() - t0:.1f}s "
                    f"-> {self.repo_id} (source {disp})"
                )
                if self.cfg.num_demos is not None and self._dataset_episode_count() >= self.cfg.num_demos:
                    total = self._dataset_episode_count()
                    if self.round_mode:
                        # Round full: finalize it (already durable) and roll to the
                        # next dataset. Keep converting -- never stop.
                        logger.info(f"round {self.round_idx} full ({total} episodes)")
                        self._advance_round()
                    else:
                        logger.info(
                            f"reached num_demos={self.cfg.num_demos} "
                            f"({total} episode(s) in the dataset); stopping"
                        )
                        self._stop = True
                        break

            if self.cfg.run_once and not self._ready_files():
                break

        logger.info(f"exiting: {n_ok} episode(s) converted this run")
        return n_ok


@draccus.wrap()
def _run(cfg: RealtimeConverterConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logger.info(pformat(asdict(cfg)))

    converter = RealtimeConverter(cfg)
    signal.signal(signal.SIGINT, converter.request_stop)
    signal.signal(signal.SIGTERM, converter.request_stop)
    converter.run()


def main():
    """Console-script entrypoint (`ws-real-time-converter`). Accepts both
    ``--num-demos`` and ``--num_demos`` spellings, then hands off to draccus."""
    normalize_hyphen_flags()
    _run()


if __name__ == "__main__":
    main()

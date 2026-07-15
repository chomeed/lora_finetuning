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

Usage (from ``policy_learning/``, in the ``policy_learning`` conda env):

    python -m lora_finetuning.realtime_converter \
        --ingest_dir=/data/incoming/board_handover \
        --dataset_repo_id=chomeed/board_handover \
        --dataset_root=/data/lerobot/board_handover \
        --default_task=board_handover

    python -m lora_finetuning.realtime_converter \
        --ingest_dir=/home/rllab4/workspace/chomeed/hdr_robot/data/test_demo \
        --dataset_repo_id=chomeed/test_demo \
        --dataset_root=/home/rllab4/workspace/chomeed/hdr_robot/data/test_demo_lerobot \
        --default_task=test_demo

    # convert whatever is already there and exit (catch-up / testing):
    python -m lora_finetuning.realtime_converter ... --run_once=true
"""

import hashlib
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

from .configs import RealtimeConverterConfig
from .hdf5_episode import IMAGE_NAMES, SchemaVersionMismatch, decode_frame_images, decode_jpeg, read_episode_arrays

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
        self.root = Path(cfg.dataset_root)
        # Where handled files go. Defaults are subdirs of the ingest dir, so the
        # top-level (non-recursive) glob never re-scans them.
        self.converted_dir = Path(cfg.converted_dir) if cfg.converted_dir else self.ingest_dir / "converted"
        self.failed_dir = Path(cfg.failed_dir) if cfg.failed_dir else self.ingest_dir / "failed"
        self._stop = False
        self.throttle = ThrottleController(cfg, lambda: self._stop)

    # -- lifecycle --
    def request_stop(self, *_):
        if not self._stop:
            logger.info("stop requested; will exit after the current episode")
        self._stop = True

    def _dataset_exists(self) -> bool:
        return (self.root / "meta" / "info.json").exists()

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
            return LeRobotDataset.resume(self.cfg.dataset_repo_id, root=str(self.root), **stream_kwargs)

        image_shapes = {}
        for name in IMAGE_NAMES:
            h, w, c = decode_jpeg(ep[f"observation.images.{name}"][0]).shape
            image_shapes[name] = (c, h, w)  # CHW to match ('channels','height','width')
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (ep["observation.state"].shape[1],),
                "names": ep["state_names"],
            },
            "action": {
                "dtype": "float32",
                "shape": (ep["action"].shape[1],),
                "names": ep["action_names"],
            },
        }
        for name in IMAGE_NAMES:
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": image_shapes[name],
                "names": ("channels", "height", "width"),
            }
        logger.info(f"creating dataset {self.cfg.dataset_repo_id} at {self.root}")
        return LeRobotDataset.create(
            repo_id=self.cfg.dataset_repo_id,
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

        ds = self._open_writer(ep)
        try:
            # Pre-decode JPEGs in worker threads while add_frame consumes; cv2
            # releases the GIL so decode scales across cores.
            with ThreadPoolExecutor(max_workers=self.cfg.num_decode_workers) as ex:
                in_flight: deque[tuple[int, Future]] = deque()
                prefetch = self.cfg.num_decode_workers + 2
                next_submit = 0
                while in_flight or next_submit < n_rows:
                    while next_submit < n_rows and len(in_flight) < prefetch:
                        in_flight.append((next_submit, ex.submit(decode_frame_images, ep, next_submit)))
                        next_submit += 1
                    idx, fut = in_flight.popleft()
                    ds.add_frame(
                        {
                            "observation.state": ep["observation.state"][idx],
                            "action": ep["action"][idx],
                            "task": task,
                            **fut.result(),
                        }
                    )
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
        return n_rows

    # -- main loop --
    def run(self) -> int:
        apply_process_priority(self.cfg.nice, self.cfg.ionice_idle)
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"watching {self.ingest_dir}/{self.cfg.glob} -> {self.cfg.dataset_repo_id} "
            f"({'run-once' if self.cfg.run_once else 'daemon'})"
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
                    f"converted {path.name}: {n} frames in {time.perf_counter() - t0:.1f}s (source {disp})"
                )

            if self.cfg.run_once and not self._ready_files():
                break

        logger.info(f"exiting: {n_ok} episode(s) converted this run")
        return n_ok


@draccus.wrap()
def main(cfg: RealtimeConverterConfig):
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


if __name__ == "__main__":
    main()

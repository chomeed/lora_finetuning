"""Push finished demo episodes from the robot to the workstation ingest queue.

The robot records ``episode_*.h5`` into a local directory (atomic rename from
``.partial`` -- a visible ``.h5`` is always complete). This daemon watches that
directory and rsync-over-SSHes each new episode into the workstation's
``tmp_demo`` ingest dir, where ``ws-real-time-converter`` picks it up. It mirrors
the converter's "the directory IS the queue" design: a local file is pending
until it has been pushed. Once the remote rename confirms delivery the local
``.h5`` is **deleted** (pass ``--keep_local`` to move it into ``sent/`` instead).

Atomic delivery, so the converter never sees a half-uploaded file:

  1. (optional) compute a ``<name>.h5.sha256`` sidecar so the converter can
     verify the transfer,
  2. rsync the sidecar into place first, then rsync the ``.h5`` to a
     ``<name>.h5.partial`` staging name (the converter's glob ignores
     ``.partial``),
  3. ``ssh mv`` the staging name to the final ``<name>.h5`` -- one atomic
     rename that reveals the complete episode with its sidecar already present.

A failed push leaves the local file in place to retry next scan; nothing is
moved aside until the remote rename succeeds.

Deliberately stdlib-only (subprocess rsync/ssh) so it runs in the robot's ROS
env. Requires ``rsync`` and ``ssh`` on PATH.

SSH auth -- no password is entered per transfer. Set up **key-based** SSH to the
workstation once, up front, so the daemon never has to prompt:

    ssh-copy-id rllab4@workstation            # one time, enter the password once

After that every push authenticates with the key, no password. If you must use
password auth instead, wrap ssh with sshpass via ``--ssh_opts``, but key auth is
strongly preferred for an unattended daemon. Pass ``--ssh_opts="-o BatchMode=yes"``
to make a missing key fail fast rather than hang waiting for a password.

Shortcut (installed by `pip install -e .`): ``robot-demo-sending``.

    robot-demo-sending \
        --local_dir=/data/demos/board_insertion/<session> \
        --remote=rllab4@workstation --remote_dir=/data/incoming/tmp_demo

    # keep a local copy instead of deleting after send:
    robot-demo-sending ... --keep_local

    # send the current backlog and exit:
    robot-demo-sending ... --run_once

Equivalent: ``python -m lora_finetuning.robot_pc.demo_sender``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import logging
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("demo_sender")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DemoSender:
    def __init__(self, args: argparse.Namespace):
        self.a = args
        self.local_dir = Path(args.local_dir)
        self.sent_dir = Path(args.sent_dir) if args.sent_dir else self.local_dir / "sent"
        self._stop = False
        self._remote_dir_ready = False
        # ssh transport shared by rsync (-e) and the mkdir/mv control commands.
        self._ssh = ["ssh", "-p", str(args.ssh_port), *shlex.split(args.ssh_opts or "")]

    # -- lifecycle --
    def request_stop(self, *_):
        if not self._stop:
            logger.info("stop requested; will exit after the current file")
        self._stop = True

    # -- remote helpers --
    def _ssh_run(self, remote_cmd: str) -> None:
        subprocess.run([*self._ssh, self.a.remote, remote_cmd], check=True)

    def _ensure_remote_dir(self) -> None:
        if not self._remote_dir_ready:
            self._ssh_run(f"mkdir -p {shlex.quote(self.a.remote_dir)}")
            self._remote_dir_ready = True

    def _rsync(self, src: Path, remote_name: str) -> None:
        dest = f"{self.a.remote}:{self.a.remote_dir}/{remote_name}"
        cmd = ["rsync", "-a", "--partial", "-e", " ".join(shlex.quote(x) for x in self._ssh)]
        if self.a.bwlimit:
            cmd.append(f"--bwlimit={self.a.bwlimit}")
        if self.a.dry_run:
            cmd.append("--dry-run")
        cmd += [str(src), dest]
        subprocess.run(cmd, check=True)

    # -- scanning --
    def _ready_files(self) -> list[Path]:
        """Complete, pending episode files directly under local_dir, oldest first.

        Top-level only, files only: the recorder keeps rejected rollouts in a
        subdirectory of this same dir (e.g. ``failure/``), and --keep_local moves
        sent files into ``sent/`` -- neither must ever be picked up. We match
        against the immediate children (fnmatch on the name), so this holds even
        if --glob contains a ``**`` that Path.glob would otherwise recurse on."""
        now = time.time()
        out = []
        for p in sorted(self.local_dir.iterdir()):
            if not p.is_file():
                continue  # skip failure/ , sent/ , converted/ ... subdirs
            if not fnmatch.fnmatch(p.name, self.a.glob):
                continue
            if p.suffix != ".h5" or p.name.endswith(".partial"):
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            if now - st.st_mtime < self.a.min_age_s:
                continue  # too fresh; let the recorder's rename settle
            out.append(p)
        return out

    # -- one push --
    def _send_one(self, path: Path) -> None:
        remote_final = path.name  # episode_x.h5
        remote_partial = remote_final + ".partial"
        self._ensure_remote_dir()

        # 1. sidecar first, so it is already present the instant the .h5 appears.
        sidecar = None
        if self.a.checksum:
            sidecar = path.with_name(path.name + ".sha256")
            sidecar.write_text(f"{_sha256(path)}  {path.name}\n")
            self._rsync(sidecar, remote_final + ".sha256")

        # 2. body to a staging name the converter's glob ignores.
        self._rsync(path, remote_partial)

        # 3. atomic reveal.
        if not self.a.dry_run:
            self._ssh_run(
                f"mv {shlex.quote(self.a.remote_dir + '/' + remote_partial)} "
                f"{shlex.quote(self.a.remote_dir + '/' + remote_final)}"
            )

        # 4. delivery confirmed -> reclaim local space. Default: delete the
        #    source (and its sidecar). With --keep_local, move them into sent/
        #    instead so the robot retains a copy. Either way the file leaves
        #    the scan set so it isn't re-sent.
        if self.a.dry_run:
            return
        if self.a.keep_local:
            self.sent_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(self.sent_dir / path.name))
            if sidecar and sidecar.exists():
                shutil.move(str(sidecar), str(self.sent_dir / sidecar.name))
        else:
            path.unlink()
            if sidecar and sidecar.exists():
                sidecar.unlink()

    # -- main loop --
    def run(self) -> int:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"watching {self.local_dir}/{self.a.glob} -> "
            f"{self.a.remote}:{self.a.remote_dir} "
            f"({'run-once' if self.a.run_once else 'daemon'}{', dry-run' if self.a.dry_run else ''})"
        )
        n_ok = 0
        while not self._stop:
            batch = self._ready_files()
            if not batch:
                if self.a.run_once:
                    break
                time.sleep(self.a.poll_interval_s)
                continue
            for path in batch:
                if self._stop:
                    break
                try:
                    t0 = time.perf_counter()
                    self._send_one(path)
                except (OSError, subprocess.CalledProcessError) as e:
                    logger.error(f"FAILED {path.name}: {e} (will retry next scan)")
                    continue
                n_ok += 1
                logger.info(f"sent {path.name} in {time.perf_counter() - t0:.1f}s")
            if self.a.run_once and not self._ready_files():
                break
        logger.info(f"exiting: {n_ok} episode(s) sent this run")
        return n_ok


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--local_dir", required=True, help="dir the robot records episode_*.h5 into")
    p.add_argument("--remote", required=True, help="ssh target for the workstation, e.g. user@host")
    p.add_argument("--remote_dir", required=True, help="workstation ingest dir (the converter's queue)")
    p.add_argument("--glob", default="episode_*.h5", help="which local files are episodes")
    p.add_argument("--ssh_port", type=int, default=22)
    p.add_argument("--ssh_opts", default="", help="extra ssh options, e.g. '-i ~/.ssh/id_ed25519'")
    p.add_argument("--bwlimit", default="", help="rsync --bwlimit (KB/s), e.g. 50000; empty = unlimited")
    p.add_argument("--poll_interval_s", type=float, default=2.0)
    p.add_argument("--min_age_s", type=float, default=1.0, help="ignore files touched more recently")
    p.add_argument("--keep_local", action="store_true",
                   help="keep a local copy (move to sent/) instead of deleting after send")
    p.add_argument("--sent_dir", default="", help="with --keep_local, move here; empty = <local_dir>/sent")
    p.add_argument("--run_once", action="store_true", help="send the backlog and exit")
    p.add_argument("--dry_run", action="store_true", help="rsync --dry-run; don't rename or move aside")
    checksum = p.add_mutually_exclusive_group()
    checksum.add_argument("--checksum", dest="checksum", action="store_true", default=True,
                          help="send a .sha256 sidecar so the converter verifies (default)")
    checksum.add_argument("--no_checksum", dest="checksum", action="store_false")
    return p.parse_args(argv)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    if shutil.which("rsync") is None or shutil.which("ssh") is None:
        logger.error("rsync and ssh must be on PATH")
        sys.exit(2)
    sender = DemoSender(args)
    signal.signal(signal.SIGINT, sender.request_stop)
    signal.signal(signal.SIGTERM, sender.request_stop)
    sender.run()


if __name__ == "__main__":
    main()

# robot_pc

The robot (Orin). Pushes finished demo/dagger episodes to the workstation's
ingest queue as they are recorded.

| Shortcut | Module | Does |
|---|---|---|
| `robot-demo-sending` | `robot_pc/demo_sender.py` | rsync-over-SSH each finished `episode_*.h5` into the workstation's `tmp_demo` ingest dir, atomically |

Deliberately **stdlib-only** (subprocess `rsync`/`ssh`) so it runs in the robot's
ROS env with no torch/lerobot/draccus. Delivery is atomic (sidecar + `.partial`
staging + remote `mv`) so the converter never sees a half-uploaded file. After a
confirmed push the local `.h5` is **deleted** (`--keep_local` moves it to `sent/`
instead).

```bash
robot-demo-sending \
    --local_dir=/data/demos/board_insertion/<session> \
    --remote=rllab4@workstation --remote_dir=/data/incoming/tmp_demo
```

**SSH auth** — set up passwordless key auth once so the daemon never prompts:

```bash
ssh-copy-id rllab4@workstation      # one time; enter the password once
```

Point `--local_dir` at the recorder's **session directory** (the recorder keeps
failures in a `failure/` subdir, which the top-level glob skips, so only kept
successes are sent).

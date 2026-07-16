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
    --remote=workstation --remote_dir=/data/incoming/tmp_demo
```

Point `--local_dir` at the recorder's **session directory** (the recorder keeps
failures in a `failure/` subdir, which the top-level scan skips, so only kept
successes are sent). Add `--run_once` to send the current backlog and exit;
`--keep_local` to move sent files to `sent/` instead of deleting them.

## First-time setup (on the robot)

**1. Install** (registers the `robot-demo-sending` shortcut; stdlib-only, no heavy deps):

```bash
pip install -e .        # from the lora_finetuning dir
```

**2. Passwordless SSH + a `workstation` alias** so the daemon never prompts and you
can use a short host name. Create a key if you don't have one, add an SSH alias, then
copy the key:

```bash
ssh-keygen -t ed25519 -C "robot"          # press Enter at every prompt (no passphrase)

mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF'

Host workstation
    HostName 169.254.186.74               # the workstation's IP (`hostname -I` on it)
    User rllab4
    IdentityFile ~/.ssh/id_ed25519
EOF
chmod 600 ~/.ssh/config

ssh-copy-id workstation                   # installs the key — enter the password once
ssh workstation 'hostname'                # passwordless test
```

With the alias, `--remote=workstation` carries the user; no `rllab4@` needed. On a
direct link-local (`169.254.*`) connection the IP can change on reconnect — update
the one `HostName` line if it stops resolving.

## Example run

```bash
# dry-run first: does the rsync but skips the remote rename + local delete
robot-demo-sending \
    --local_dir=/root/demo_data/my_experiment \
    --remote=workstation \
    --remote_dir=/home/rllab4/workspace/chomeed/hdr_robot/policy_learning/lora_finetuning/tests \
    --run_once --dry_run

# for real: drop --dry_run (keep --run_once to send once and exit,
# or drop it too to run as a daemon that keeps watching)
robot-demo-sending \
    --local_dir=/root/demo_data/my_experiment \
    --remote=workstation \
    --remote_dir=/home/rllab4/workspace/chomeed/hdr_robot/policy_learning/lora_finetuning/tests
```

The user behind `--remote` (here `rllab4`) needs write permission to `--remote_dir`;
the sender creates it with `ssh mkdir -p` if missing.

# lora_finetuning

On-the-fly LoRA fine-tuning of a base policy (PI05), served through LeRobot's async
inference stack **without restarting the policy server**.

Two processes connected by gRPC — they can run on different machines, no shared
filesystem required. It's a **pull** model: the client asks for params when it wants them.

```
trainer.py            --serve_port=8090      trains LoRA adapters on a LeRobot dataset,
(hosts AdapterService)                       keeps the latest version in memory
                              ▲
      GetLatestAdapter(have_version=N)  │   client asks on its own schedule
      ← latest adapter if newer, else   │   (empty reply when nothing new → cheap)
        empty stream                    ▼
lora_policy_server.py --adapter_addr=HOST:8090   pulls the latest adapter, materializes it
(pulls from the trainer)                         into a local temp dir, swaps it in-place at
                                                 action-chunk boundaries (or on handshake)
```

Each version is ~5 MB (adapter only, never the 4.1B base): chunked safetensors +
`adapter_config.json`, inline. The trainer just holds the current adapter; **it never
pushes** — bytes move only when the client requests and a newer version exists.

LeRobot's own `policy_server.py` is **not modified**. `LoRAPolicyServer` subclasses
`PolicyServer` and overrides two methods.

## Why this shape

This is the base-policy fine-tuning leg of [EXPO](https://arxiv.org/abs/2507.07986).
EXPO keeps a large expressive base policy and a lightweight edit/residual policy on
top; the residual is trained with RL, and the base is *continuously fine-tuned with a
plain imitation objective* on the replay buffer. No value weighting, no backprop from
Q into the base. The improvement comes from *which* actions end up in the data — if
you train on value-improved (edited / corrected / successful) rollouts, the base
distills toward them and the residual has less work to do over time.

So the trainer here is deliberately dumb: it runs PI05's own flow-matching loss on a
LeRobot dataset. All the filtering lives upstream, in **which episodes you point it
at**. LoRA is what makes it practical: the trainable set is ~1.35M params (0.03% of
the base), so a version is ~5 MB — small enough to hand to a live inference server
between action chunks, where a full 4.1B state dict never would be.

## Quickstart

Both processes need the `policy_learning` conda env (it has `peft`, `torch`, and
`lerobot` installed editable). Run from the `policy_learning/` directory.

**1. Start the server** (drop-in replacement for `lerobot.async_inference.policy_server`):

```bash
python -m lora_finetuning.lora_policy_server \
    --host=0.0.0.0 --port=8080 --fps=30 \
    --adapter_addr=<trainer-host>:8090 \
    --reload_on=chunk
```

It accepts every stock `PolicyServerConfig` flag. The robot client connects and sends
`SendPolicyInstructions` exactly as before — the base policy still loads from the
checkpoint path the client names. Omit `--adapter_addr` to run as a plain
`PolicyServer` with no hot-reload.

**2. Start the trainer**, pointed at the *same base checkpoint the client will ask the
server to load*. It hosts the AdapterService on `--serve_port`:

```bash
python -m lora_finetuning.trainer \
    --pretrained_path=outputs/train/<run>/pretrained_model \
    --dataset_repo_id=chomeed/<dataset> \
    --dataset_root=/path/to/dataset \
    --serve_host=0.0.0.0 --serve_port=8090 \
    --steps=20000 --batch_size=8 --lr=1e-4 --publish_freq=500
```

The trainer streams an identity adapter at step 0 (LoRA is initialized with `B=0`,
so it provably does not change behavior), which lets the server inject the adapter
layers once, up front. Every publish after that is a cheap in-place weight swap. When
training finishes the trainer keeps serving the final adapter (Ctrl-C to exit), so a
policy server can still subscribe afterward.

Watch the server log for:

```
LoRA adapter v1 injected (trainer step 0, loss nan) in 412ms
LoRA adapter v2 swapped (trainer step 500, loss 0.0431) in 38ms
```

## Feeding it live: the realtime converter

`realtime_converter.py` is the workstation leg of the data loop. The robot uploads
finished `episode_*.h5` files into an ingest directory (atomic rename from
`.partial`, so a visible `.h5` is always complete); the converter watches that
directory and **appends each new episode to a single growing LeRobot dataset** —
the same dataset the trainer trains on. Point the trainer at the same
`--dataset_root`/`--dataset_repo_id` and it picks up the new episodes on its next
(re)start.

```bash
python -m lora_finetuning.realtime_converter \
    --ingest_dir=/data/incoming/board_handover \
    --dataset_repo_id=chomeed/board_handover \
    --dataset_root=/data/lerobot/board_handover \
    --default_task=board_handover
```

It is built to run next to a live, high-priority policy server:

- **It yields.** The process is CPU-niced (`--nice`, default 15) and, best-effort,
  IO-idle (`--ionice_idle`). More importantly it *pauses whole-episode conversion*
  whenever the GPU is busy — `nvidia-smi` utilization at/above `--gpu_util_pause`
  (default 60%), releasing only once it drops to `--gpu_util_resume` (40%, so a
  server hovering near the line doesn't cause start/stop churn) — or whenever a
  pause-file exists (`touch <ingest_dir>/.pause` to stop, `rm` to resume). If
  `nvidia-smi` can't be read the GPU signal is ignored and the pause-file still
  works. JPEG-decode and AV1-encode (the expensive parts) run only when clear.
- **It is crash-safe and idempotent — the ingest dir is the queue.** A file
  sitting in the ingest dir is pending; there is no ledger. Each episode is
  converted, then `finalize()`d into a valid, readable checkpoint, and only
  *then* is the source `.h5` (and its `.sha256` sidecar) **deleted** — the robot
  keeps the raw recording. A crash mid-episode leaves the source in place, so it
  is simply reconverted next run; nothing to keep in sync or lose. Set
  `--delete_on_success=false` to move converted files into `ingest_dir/converted/`
  instead of deleting them. Permanently bad episodes (unreadable, checksum
  mismatch, schema/shape mismatch) are moved to `ingest_dir/failed/` so they
  aren't retried on every scan rather than wedging the loop.
- **It verifies.** If the robot drops a `<name>.sha256` sidecar next to an
  episode, the converter checks it before converting (`--verify_checksum`, on by
  default; a no-op when no sidecar is present).

- **It streams the video encode.** With `--streaming_encoding` (default on) each
  frame is piped into the AV1 encoder as it's decoded and added, instead of
  writing every frame as a temp PNG and batch-encoding the whole episode at save
  time. That removes the temp-image disk churn entirely (no lingering PNGs) and
  spreads the encode CPU across the episode rather than bursting it at the end —
  both of which matter when you're sharing the box with inference. On the sample
  720p episodes this was ~6.5 s/episode vs ~56 s for the batch path. `--vcodec`
  (default `libsvtav1`), `--encoder_queue_maxsize`, and `--encoder_threads` tune
  it; set `--streaming_encoding=false` to fall back to batch encoding. Streaming
  needs a system ffmpeg with the AV1 encoder built in (see `render_episodes.py`
  in `lerobot_chomeed_datasets/` for the same requirement on the read side).

`--run_once=true` converts the current backlog and exits (catch-up / testing);
the default is to keep watching. Ctrl-C / SIGTERM finishes the current episode's
`finalize()`, writes the ledger, and exits cleanly. See `RealtimeConverterConfig`
in `configs.py` for the full flag list (`--poll_interval_s`, `--num_decode_workers`,
`--task_filter`, `--glob`, etc.).

The episode-reading half is a self-contained copy of the Orin-side offline
converter (`orin_demo_collection/convert_to_lerobot.py`), vendored into
`hdf5_episode.py` so this workstation package needs neither the ROS package nor
its env. Keep the two in sync if the recording schema changes.

## The base checkpoint must match

The trainer and the server must load the **same** `pretrained_path`. An adapter is a
delta against specific base weights; applying it to a different base is silently wrong,
not an error. The server takes its base path from the robot client's
`SendPolicyInstructions`, so it is the *client's* `--policy.path` that has to agree with
the trainer's `--pretrained_path`.

## When adapters get swapped in

| `--reload_on` | Behavior |
|---|---|
| `chunk` (default) | Pull at every action-chunk boundary. Fastest propagation. The policy can change mid-episode. |
| `handshake` | Pull only when a client connects/reconnects, i.e. between episodes. |

`--reload_on` is really *when the client pulls*. Each boundary it calls
`GetLatestAdapter(have_version=loaded)`; if the trainer has nothing newer the reply is an
empty stream — one small round-trip, no bytes transferred — so pulling every boundary is
cheap. Only when a new version exists does it download and swap. The swap happens on the
inference thread, before the observation is preprocessed, so it never lands in the middle
of a forward pass, and the client reassembles the whole adapter into a local dir before
applying it, so a half-received adapter is never applied. If a fetch fails the client
returns nothing (rebuilding its channel next time) and the server keeps its current
weights; if an adapter fails to load, same thing — it logs and keeps serving.

Use `handshake` if PI05's real-time chunking (RTC) blending across a weight change
worries you, or on hardware where a mid-rollout behavior change is unacceptable.

## LoRA targets (read this before changing them)

Default targets for PI05 (`PI05_LORA_TARGETS` in `configs.py`) — 40 modules, ~1.35M
params at `r=16`:

- the action expert's `self_attn.q_proj` / `v_proj` (18 layers × 2)
- `model.action_in_proj`, `model.action_out_proj`
- `model.time_mlp_in`, `model.time_mlp_out`

The SigLIP vision tower and the PaliGemma trunk stay frozen.

**We deliberately do not use LeRobot's `PI05Policy._get_default_peft_targets()`.** That
regex is stale: it targets `model.state_proj` (PI05 has no such module — state enters as
prefix tokens, not a projection) and `model.action_time_mlp_{in,out}` (PI0's names; PI05
calls them `model.time_mlp_{in,out}`). Neither ever matches, so the stock default
silently trains the time MLPs not at all. Verified against a real PI05 checkpoint: the
stock regex hits 38 modules, ours hits 40.

Override with `--lora.target_modules=...` (regex or list), `--lora.r=32`, etc.

## Config reference

Trainer (`--help` for the full list):

| Flag | Default | Note |
|---|---|---|
| `--pretrained_path` | required | base checkpoint; must match what the server loads |
| `--dataset_repo_id` / `--dataset_root` | required / None | LeRobot dataset |
| `--serve_host` / `--serve_port` | `0.0.0.0` / 8090 | where the AdapterService binds |
| `--episodes` | all | subset to train on — this is your value filter |
| `--publish_freq` | 500 | steps between adapter publishes |
| `--lora.r` / `--lora.lora_alpha` | 16 / 32 | |
| `--gradient_checkpointing` | true | ~30% slower steps, much less VRAM |
| `--use_amp` | true | bf16 autocast |
| `--resume_adapter_path` | None | continue training a published adapter |

Server: every stock `PolicyServerConfig` flag, plus `--adapter_addr`, `--reload_on`, and
`--adapter_cache_dir` (where received adapters land; defaults to a temp dir). Omitting
`--adapter_addr` makes it behave exactly like the stock `PolicyServer`.

## Files

| File | Purpose |
|---|---|
| `trainer.py` | flow-matching BC on LoRA adapters; serves the latest version on request |
| `realtime_converter.py` | watches an ingest dir, appends new HDF5 episodes to the growing LeRobot dataset the trainer reads; GPU/pause-file throttled, deletes each source `.h5` once converted (ingest dir is the queue) |
| `hdf5_episode.py` | reads one recorded HDF5 episode into arrays (vendored from the Orin-side offline converter) |
| `lora_policy_server.py` | `PolicyServer` subclass that pulls + hot-reloads adapters |
| `example_client.py` | standalone demo client: load a base policy, pull adapters, no robot |
| `transport/` | gRPC hand-off: `service.proto`, `wire.py` (serialize/chunk/reassemble), `publisher.py` (trainer server), `client.py` (pull client), `applier.py` (inject/swap into a live policy) |
| `configs.py` | draccus configs + the PI05 LoRA targets |

Regenerate the gRPC stubs after editing `transport/service.proto` (run from
`policy_learning/`):

```bash
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. \
    lora_finetuning/transport/service.proto
```

## Verified

- PEFT injects LoRA in place, so the server's existing `self.policy` calls route through
  the adapter with no other changes to `PolicyServer`.
- A freshly published adapter is an exact identity (B=0) — publishing at step 0 is safe.
- After a hot swap, the server reproduces the trainer's policy output bit-for-bit.
- The gRPC round-trip is byte-exact: the safetensors the subscriber writes to disk match
  what the trainer serialized (loopback test in `transport/`).
- The PI05 target regex matches 40 modules on a real checkpoint (1.35M params, 5.4 MB).

Not yet run end-to-end against a live robot + a real training run — the loop above has
been exercised with a stand-in model, not PI05 on GPU.

## Known gaps

- **Stale actions.** If you later add a residual/edit policy on top (full EXPO), note
  that a residual is defined *relative to* the base's output. Once the base starts
  moving, any `base_action` cached in a replay buffer goes stale. Recompute it from the
  live base, or version-tag transitions.
- No advantage/Q weighting. Filtering is by episode selection only.
- Single-process trainer; no distributed training.

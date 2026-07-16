# lora_finetuning

On-the-fly LoRA fine-tuning of a base policy (PI05), served through LeRobot's async
inference stack **without restarting the policy server** — wired into a live DAgger
(SIRIUS) loop: the robot streams corrected rollouts to the workstation, which
converts them, trains a LoRA adapter on them, and hot-swaps it into the running
policy.

## Layout (by machine)

```
robot_pc/         the Orin — pushes recorded episodes to the workstation queue
workstation_pc/   rllab4 — policy server + realtime HDF5→LeRobot converter + LoRA trainer
common/           shared library: configs, trainer/policy-server cores, obs/action
                  schema, dagger manifest, and the gRPC adapter transport
```

The **same** trainer/policy-server code runs on a remote GPU box too — bind
`--serve_host=0.0.0.0` instead of loopback; there is no separate remote package.
Install once (registers the console-script shortcuts): from this directory,

```bash
pip install -e .
```

| Shortcut | Runs on | What |
|---|---|---|
| `robot-demo-sending` | robot | rsync finished episodes → workstation `tmp_demo` queue |
| `ws-real-time-converter` | workstation | HDF5 → LeRobot, `--mode` projection, intervention + manifest, sirius-round rollover |
| `ws-lora-finetuning` | workstation / remote | LoRA trainer; single-shot or the `--dagger_loop` service |
| `ws-serve-policy` | workstation / remote | policy server that pulls + hot-swaps adapters |

Both hyphen and underscore flag spellings are accepted (`--num-demos` == `--num_demos`).
See [robot_pc/](robot_pc/README.md) and [workstation_pc/](workstation_pc/README.md)
for per-role usage.

## The loop

```
robot_pc:  robot-demo-sending ──rsync──▶ tmp_demo/            (the queue)
workstation:
  ws-real-time-converter  tmp_demo/ ──▶ <task>_sirius_round1, round2, …  (N eps/round)
  ws-lora-finetuning --dagger_loop  ◀── watches the rounds; each round trains a
        50/50 mix of baseline demos and dagger *intervention* transitions (SIRIUS),
        publishes an adapter over gRPC, resets the LR, waits for the next round
  ws-serve-policy  ◀── pulls the latest adapter and swaps it in at a chunk boundary
```

The adapter transport is a **pull** model: the trainer hosts an `AdapterService`
and holds the latest adapter; the policy server asks for it when it wants it
(every action-chunk boundary, or only between episodes). Each version is ~5 MB
(adapter only, never the 4.1B base): chunked safetensors + `adapter_config.json`.
`LoRAPolicyServer` subclasses LeRobot's `PolicyServer` and overrides two methods;
LeRobot's own `policy_server.py` is not modified.

## Why this shape

This is the base-policy fine-tuning leg of [EXPO](https://arxiv.org/abs/2507.07986):
keep a large expressive base policy and continuously fine-tune it with a plain
imitation objective on the improving replay buffer. The trainer is deliberately
dumb — it runs PI05's own flow-matching loss on a LeRobot dataset. All the value
filtering lives upstream, in **which frames end up in the data**: here that's the
SIRIUS weighting (`common/schema.py` picks the policy's channels; the DAgger loop
weights human-intervention frames 50/50 against baseline demos). LoRA is what makes
it practical: ~1.35M trainable params (0.03% of the base), so a version is small
enough to hand to a live inference server between action chunks.

## Data: the SIRIUS round datasets

The converter writes `<task>_sirius_round<N>` LeRobot datasets and rolls to the
next every `--num_demos` episodes (continuous — it never stops). Every converted
episode carries:

- **`--mode` projection** — the recording is always the full rig (41-D state /
  19-D action, head + both wrists); the converter projects it down to the policy's
  I/O schema (`insertion_15`, `handover`, …; `full` = no projection). See
  [common/schema.py](common/schema.py) — every mode's channels must be a subset of
  the full layout, so projection is a pure index select.
- **a per-frame `intervention` feature** — read from the episode's `/intervention`
  dataset (1 = human took over). The DAgger trainer uses only these frames from the
  dagger side of the mix.
- **a `meta/dagger_manifest.jsonl` row** — episode index, task, policy id
  (`--dagger_policy` or an episode attr), intervention count, and the live adapter
  version + trainer step (from `ws-serve-policy --version_status_file`, read via the
  converter's `--policy_status_file`). This is how the workstation knows which
  policy version each episode was collected under.

## The base checkpoint must match

The trainer and the server must load the **same** `pretrained_path`. An adapter is a
delta against specific base weights; applying it to a different base is silently
wrong, not an error. Normalization always reuses the baseline checkpoint's stats
(the trainer normalizes with the checkpoint's preprocessor, and the SIRIUS mix
freezes to the baseline demo dataset's stats) — dataset stats are never recomputed.

## LoRA targets (read before changing) and variants

Default PI05 targets (`PI05_LORA_TARGETS` in `common/configs.py`) — 40 modules,
~1.35M params at `r=16`: the action expert's `self_attn.q_proj`/`v_proj` (18×2),
`model.action_in_proj`/`action_out_proj`, and `model.time_mlp_in`/`time_mlp_out`.
The SigLIP tower and PaliGemma trunk stay frozen. We deliberately **do not** use
LeRobot's `_get_default_peft_targets()` — its regex is stale (targets
`model.state_proj`, which PI05 doesn't have, and PI0's `action_time_mlp_*` names),
so it silently skips the time MLPs. Verified: the stock regex hits 38 modules, ours
hits 40.

`--lora_variant small|medium|big` sets `lora.r`/`lora.lora_alpha` (4/8, 16/32,
32/64) — use `small` to iterate quickly.

## Files

| File | Purpose |
|---|---|
| `common/trainer.py` | flow-matching BC on LoRA; single-shot + the DAgger round loop; serves adapters |
| `common/policy_server.py` | `PolicyServer` subclass that pulls + hot-reloads adapters; writes the version status file |
| `common/schema.py` | observation/action `MODE_SCHEMAS` + full-rig `STATE_KEYS`/`ACTION_KEYS` + projection |
| `common/hdf5_episode.py` | reads one recorded HDF5 episode (incl. `/intervention` + policy-version attrs) |
| `common/manifest.py` | dagger manifest read/write + policy-version status file |
| `common/configs.py` | draccus configs + PI05 LoRA targets + LoRA variants |
| `common/transport/` | gRPC adapter hand-off: `service.proto`, `wire.py`, `publisher.py`, `client.py`, `applier.py` |
| `workstation_pc/realtime_converter.py` | ingest daemon (mode, intervention, manifest, sirius rounds; GPU/pause throttled) |
| `workstation_pc/{train,serve_policy}.py` | thin entrypoints over `common` |
| `robot_pc/demo_sender.py` | stdlib-only rsync-over-SSH push into the workstation queue |

Regenerate the gRPC stubs after editing `common/transport/service.proto` (from
`policy_learning/`):

```bash
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. \
    lora_finetuning/common/transport/service.proto
```

## Converter internals (co-locating with a live policy server)

The converter is built to run next to a high-priority policy server:

- **It yields.** CPU-niced (`--nice`), best-effort IO-idle (`--ionice_idle`), and it
  *pauses* whole-episode conversion whenever the GPU is busy (`nvidia-smi`
  utilization ≥ `--gpu_util_pause`, resuming at `--gpu_util_resume` — hysteresis) or
  a pause-file exists (`touch <ingest_dir>/.pause`).
- **The ingest dir is the queue.** No ledger: a visible `.h5` is pending; each is
  converted, `finalize()`d into a valid checkpoint, and only then is the source
  deleted (`--delete_on_success=false` moves it to `converted/` instead).
  Unconvertible episodes move to `failed/`.
- **It verifies + streams the encode.** Optional `<name>.sha256` sidecar check;
  streaming AV1 encode (`--streaming_encoding`, needs a system ffmpeg with
  `libsvtav1`) pipes frames as they decode instead of buffering temp PNGs.

The HDF5 read half (`common/hdf5_episode.py`) is vendored from the Orin-side
offline converter (`orin_demo_collection/convert_to_lerobot.py`); keep them in sync
if the recording schema changes.

## Verified

- End-to-end on real dagger HDF5: `--mode` projection (e.g. `insertion_15` → 15-D
  state / 8-D action, head + wrists), per-frame `intervention` feature written to
  the parquet, per-episode manifest rows, and sirius-round rollover
  (round1 fills → round2 …) with resume-at-partial-round on restart.
- PEFT injects LoRA in place; a step-0 adapter is an exact identity (B=0); after a
  hot swap the server reproduces the trainer's output bit-for-bit; the gRPC
  round-trip is byte-exact.
- Not yet run end-to-end against a live robot + a real GPU training run.

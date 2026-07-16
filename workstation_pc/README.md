# workstation_pc

The box next to the robot (rllab4). Serves the policy, ingests demos into LeRobot
datasets, and trains the LoRA adapter. **The trainer and policy-server entrypoints
also run on a remote GPU box** — just bind `--serve_host=0.0.0.0` there instead of
loopback; there is no separate remote package.

| Shortcut | Module | Does |
|---|---|---|
| `ws-real-time-converter` | `realtime_converter.py` | watch the ingest dir; append each HDF5 episode to a LeRobot dataset, projected to `--mode`, tagged with the intervention flag + a dagger manifest |
| `ws-lora-finetuning` | `train.py` → `common.trainer` | LoRA trainer; single-shot or the DAgger round service (`--dagger_loop`) |
| `ws-serve-policy` | `serve_policy.py` → `common.policy_server` | adapter-hot-reloading policy server |

Both hyphen and underscore flag spellings work (`--num-demos` == `--num_demos`).

## The dagger loop

```
robot_pc:  robot-demo-sending ──rsync──▶ tmp_demo/            (queue)
workstation:
  ws-real-time-converter  tmp_demo/ ──▶ <task>_sirius_round1, round2, …   (30 eps/round)
  ws-lora-finetuning --dagger_loop  ◀── watches those rounds; trains 50/50
                       baseline-demo / dagger-intervention (SIRIUS), publishes
                       an adapter each round, resets LR, waits for the next round
  ws-serve-policy  ◀── pulls + hot-swaps the adapter; writes its live version to
                       --version_status_file, which the converter tags episodes with
                       (both default to /tmp/lora_policy_version.json — wired
                       automatically when everything runs on one machine)
```

### Converter — SIRIUS round mode

```bash
ws-real-time-converter \
    --ingest_dir=/data/incoming/tmp_demo \
    --dagger_datasets_dir=/data/lerobot/dagger \
    --default_task=board_insertion --mode=insertion_15 \
    --num-demos 30
```

`--policy_status_file` defaults to the server's `--version_status_file` path
(`/tmp/lora_policy_version.json`), so adapter-version tagging works with no
extra flags; pass `--policy_status_file=""` to disable.

Writes `board_insertion_sirius_round1`, and every `--num_demos` episodes rolls
over to `…_round2`, `…_round3`, … continuously. `--mode` projects the full-rig
recording to the policy's I/O schema (see [common/schema.py](../common/schema.py)).
Each episode carries a per-frame `intervention` feature and a row in that round's
`meta/dagger_manifest.jsonl` (policy id + adapter version + intervention count).
Omit `--dagger_datasets_dir` for single-dataset mode.

### Trainer — DAgger service

```bash
ws-lora-finetuning \
    --pretrained_path=/home/rllab4/workspace/chomeed/hdr_robot/policy_learning/outputs/sirius/board_insertion_pi05 \
    --dagger_loop=true \
    --baseline_dataset_repo_id=chomeed/board_insertion_ablation_head_fixed_quantile_k30_relative_action \
    --dagger_datasets_dir=/home/rllab4/workspace/chomeed/hdr_robot/policy_learning/lora_finetuning/tests3_lerobot \
    --dataset_repo_id=unused-in-dagger-mode \
    --serve_host=127.0.0.1 --serve_port=8090 --publish-freq 50 \
    --lora_variant small
```

Each round: rebuild a `SIRIUSDataset` mix of the baseline demos (all frames) +
every `*_sirius_round*` dataset's **intervention transitions** at
`p_intv=0.5`, train `--publish_freq` steps, publish the adapter, reset the LR +
scheduler, then wait for new dagger episodes/rounds. Normalization reuses the
baseline checkpoint's stats (never recomputed). `--lora_variant small|medium|big`
for quick tests vs full runs. `--steps<=0` runs forever.

### Serve

```bash
ws-serve-policy --host=0.0.0.0 --port=8080 --fps=30 \
    --adapter_addr=127.0.0.1:8090 --reload_on=chunk
```

The server writes its live adapter version to `--version_status_file`
(default `/tmp/lora_policy_version.json`; version 0 = base policy at startup).
The console shows lifecycle events only (startup, adapter swaps, errors);
pass `--verbose=true` for the stock per-observation inference logs.

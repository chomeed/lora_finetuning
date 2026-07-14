# lora_finetuning

On-the-fly LoRA fine-tuning of a base policy (PI05), served through LeRobot's async
inference stack **without restarting the policy server**.

Two processes with one directory between them:

```
learner.py            --adapter_dir=DIR    trains LoRA adapters on a LeRobot dataset,
                                           publishes a new version every N steps
                              │
                              ▼  ~5 MB per version (adapter only, never the 4.1B base)
                            DIR/
                              latest.json
                              v000007/{adapter_config.json, adapter_model.safetensors}
                              │
                              ▼
lora_policy_server.py --adapter_dir=DIR    serves the policy, swaps adapters in-place
                                           at action-chunk boundaries
```

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

So the learner here is deliberately dumb: it runs PI05's own flow-matching loss on a
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
    --adapter_dir=/path/to/adapters \
    --reload_on=chunk
```

It accepts every stock `PolicyServerConfig` flag. The robot client connects and sends
`SendPolicyInstructions` exactly as before — the base policy still loads from the
checkpoint path the client names.

**2. Start the learner**, pointed at the *same base checkpoint the client will ask the
server to load*, and the same adapter dir:

```bash
python -m lora_finetuning.learner \
    --pretrained_path=outputs/train/<run>/pretrained_model \
    --dataset_repo_id=chomeed/<dataset> \
    --dataset_root=/path/to/dataset \
    --adapter_dir=/path/to/adapters \
    --steps=20000 --batch_size=8 --lr=1e-4 --publish_freq=500
```

The learner publishes an identity adapter at step 0 (LoRA is initialized with `B=0`,
so it provably does not change behavior), which lets the server inject the adapter
layers once, up front. Every publish after that is a cheap in-place weight swap.

Watch the server log for:

```
LoRA adapter v1 injected (learner step 0, loss nan) in 412ms
LoRA adapter v2 swapped (learner step 500, loss 0.0431) in 38ms
```

## The base checkpoint must match

The learner and the server must load the **same** `pretrained_path`. An adapter is a
delta against specific base weights; applying it to a different base is silently wrong,
not an error. The server takes its base path from the robot client's
`SendPolicyInstructions`, so it is the *client's* `--policy.path` that has to agree with
the learner's `--pretrained_path`.

## When adapters get swapped in

| `--reload_on` | Behavior |
|---|---|
| `chunk` (default) | At any action-chunk boundary. Fastest propagation. The policy can change mid-episode. |
| `handshake` | Only when a client connects/reconnects, i.e. between episodes. |

The swap happens on the inference thread, before the observation is preprocessed, so it
never lands in the middle of a forward pass. Publishing is atomic (rename-into-place),
so the server never reads a half-written adapter. If an adapter fails to load, the
server logs the error and keeps serving the weights it already has.

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

Learner (`--help` for the full list):

| Flag | Default | Note |
|---|---|---|
| `--pretrained_path` | required | base checkpoint; must match what the server loads |
| `--dataset_repo_id` / `--dataset_root` | required / None | LeRobot dataset |
| `--adapter_dir` | required | watch dir the server polls |
| `--episodes` | all | subset to train on — this is your value filter |
| `--publish_freq` | 500 | steps between adapter publishes |
| `--keep_last` | 5 | versioned dirs retained on disk |
| `--lora.r` / `--lora.lora_alpha` | 16 / 32 | |
| `--gradient_checkpointing` | true | ~30% slower steps, much less VRAM |
| `--use_amp` | true | bf16 autocast |
| `--resume_adapter_path` | None | continue training a published adapter |

Server: every stock `PolicyServerConfig` flag, plus `--adapter_dir` and `--reload_on`.
Omitting `--adapter_dir` makes it behave exactly like the stock `PolicyServer`.

## Files

| File | Purpose |
|---|---|
| `learner.py` | flow-matching BC on LoRA adapters; publishes versions |
| `lora_policy_server.py` | `PolicyServer` subclass with adapter hot-reload |
| `adapter_store.py` | the learner↔server hand-off protocol (atomic publish, cheap poll) |
| `configs.py` | draccus configs + the PI05 LoRA targets |

## Verified

- PEFT injects LoRA in place, so the server's existing `self.policy` calls route through
  the adapter with no other changes to `PolicyServer`.
- A freshly published adapter is an exact identity (B=0) — publishing at step 0 is safe.
- After a hot swap, the server reproduces the learner's policy output bit-for-bit.
- `keep_last` pruning bounds disk use.
- The PI05 target regex matches 40 modules on a real checkpoint (1.35M params, 5.4 MB).

Not yet run end-to-end against a live robot + a real training run — the loop above has
been exercised with a stand-in model, not PI05 on GPU.

## Known gaps

- **Stale actions.** If you later add a residual/edit policy on top (full EXPO), note
  that a residual is defined *relative to* the base's output. Once the base starts
  moving, any `base_action` cached in a replay buffer goes stale. Recompute it from the
  live base, or version-tag transitions.
- No advantage/Q weighting. Filtering is by episode selection only.
- Single-process learner; no distributed training.

"""Minimal example: load a base policy and pull LoRA adapters from trainer.py on demand.

This is the client side of the transport *without* the full lerobot async-inference
policy server or a robot. It loads the base policy from a checkpoint and, on a schedule
*it* controls, calls ``AdapterClient.fetch()`` to ask the trainer for the latest adapter,
injecting/swapping whatever comes back into the live policy -- the same thing
LoRAPolicyServer does at its chunk boundaries, minus serving observations to a robot.

The point of this example is the pull: nothing arrives unless the client asks. Here it
asks every ``--request_every_s`` seconds; wire that trigger to whatever you like (a chunk
boundary, an episode reset, a keypress).

As a sanity signal it prints the total L2 norm of the LoRA ``B`` matrices after each
apply: the identity adapter published at step 0 has ``B = 0`` (norm ~0), and the norm
grows as the trainer learns -- so a rising number is proof the live policy is tracking it.

Usage (run the trainer first, then this):

    python -m lora_finetuning.example_client \
        --pretrained_path=/home/rllab4/workspace/chomeed/hdr_robot/policy_learning/outputs/ablation/board_insertion_ablation_head_pi05_delta_recomputed_stats_25k \
        --adapter_addr=127.0.0.1:8090 \
        --policy_type=pi05 --device=cuda --request_every_s=2
"""

import argparse
import logging
import time

from lerobot.policies import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins

from .transport import AdapterApplier, AdapterClient

logger = logging.getLogger("example_client")


def lora_b_norm(peft_model) -> float:
    """Total L2 norm of every LoRA ``B`` matrix -- ~0 for the identity adapter."""
    total = 0.0
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            total += p.detach().float().pow(2).sum().item()
    return total**0.5


def main():
    # force=True: lerobot's import installs a root WARNING handler, which would make a
    # plain basicConfig a no-op and swallow every INFO line below. force replaces it.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_path", required=True, help="base checkpoint (must match the trainer's)")
    parser.add_argument("--adapter_addr", required=True, help="trainer AdapterService host:port")
    parser.add_argument("--policy_type", default="pi05")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request_every_s", type=float, default=2.0, help="how often the client pulls")
    args = parser.parse_args()

    register_third_party_plugins()

    logger.info(f"Loading base {args.policy_type} policy from {args.pretrained_path}")
    policy = get_policy_class(args.policy_type).from_pretrained(args.pretrained_path)
    if not policy.config.pretrained_path:
        policy.config.pretrained_path = args.pretrained_path
    policy.to(args.device)
    policy.eval()

    applier = AdapterApplier(device=args.device)
    client = AdapterClient(args.adapter_addr, progress=True)
    logger.info(f"Will pull from {args.adapter_addr} every {args.request_every_s}s (Ctrl-C to stop)")

    try:
        while True:
            meta = client.fetch()  # the client decides when to ask; here, once per loop
            if meta is not None:
                start = time.perf_counter()
                try:
                    action = applier.apply(policy, meta.local_dir, version=meta.version)
                except Exception as e:
                    logger.error(f"Failed to apply adapter v{meta.version} from {meta.local_dir}: {e}")
                else:
                    client.mark_loaded(meta)
                    elapsed = (time.perf_counter() - start) * 1000
                    loss = meta.loss if meta.loss is not None else float("nan")
                    logger.info(
                        f"adapter v{meta.version} {action} (trainer step {meta.step}, loss {loss:.4f}) "
                        f"in {elapsed:.0f}ms | lora_B norm = {lora_b_norm(applier.peft_model):.4f}"
                    )
            time.sleep(args.request_every_s)
    except KeyboardInterrupt:
        logger.info("Stopping")
    finally:
        client.close()


if __name__ == "__main__":
    main()

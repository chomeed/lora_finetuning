"""Publisher-only 'trainer' for testing the adapter transport WITHOUT a dataset.

Loads the real base policy and wraps it in LoRA exactly as ``trainer.py`` does (same
target modules, same adapter shapes), then serves it via AdapterService. Every
``--publish_every_s`` seconds it nudges the LoRA ``B`` matrices by a small random step
and publishes a new version -- faking training progress so a client sees v1, v2, v3...
with a rising ``lora_B`` norm. No dataset, no optimizer, no gradients.

Point ``example_client.py`` (or a real ``lora_policy_server``) at it to verify the pull
transport end-to-end against your actual checkpoint.

Usage:
    python -m lora_finetuning.mock_trainer \
        --pretrained_path=/home/rllab4/workspace/chomeed/hdr_robot/policy_learning/outputs/ablation/board_insertion_ablation_head_pi05_delta_recomputed_stats_25k \
        --serve_host=0.0.0.0 --serve_port=8090 \
        --publish_every_s=3 --device=cuda
"""

import argparse
import logging
import threading

import torch

from .configs import LoRATrainerConfig
from .trainer import build_policy
from .transport import AdapterPublisher

logger = logging.getLogger("mock_trainer")


def perturb_lora_b(peft_model, step_size: float) -> None:
    """Add small Gaussian noise to every LoRA ``B`` matrix (identity B=0 -> nonzero)."""
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if p.requires_grad and "lora_B" in name:
                p.add_(torch.randn_like(p) * step_size)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,  # lerobot's import pre-installs a root WARNING handler; override it
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--serve_host", default="0.0.0.0")
    parser.add_argument("--serve_port", type=int, default=8090)
    parser.add_argument("--policy_type", default="pi05")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--publish_every_s", type=float, default=3.0)
    parser.add_argument("--step_size", type=float, default=0.02, help="LoRA B perturbation scale")
    args = parser.parse_args()

    # Reuse trainer.build_policy so the LoRA wrapping is identical to real training.
    # dataset_repo_id is required by the config but build_policy never touches it.
    cfg = LoRATrainerConfig(
        pretrained_path=args.pretrained_path,
        dataset_repo_id="<unused-by-mock>",
        policy_type=args.policy_type,
        device=args.device,
        gradient_checkpointing=False,
    )
    logger.info(f"Loading + LoRA-wrapping {args.policy_type} from {args.pretrained_path}")
    policy, peft_model = build_policy(cfg)
    peft_model.eval()

    publisher = AdapterPublisher(host=args.serve_host, port=args.serve_port)
    publisher.start()

    version = 1
    publisher.publish(peft_model, version=version, step=0)  # identity adapter (B=0)
    logger.info(f"Serving identity adapter v{version} on {args.serve_host}:{args.serve_port}")

    # Fake training: perturb + republish on a timer until Ctrl-C.
    stop = threading.Event()
    try:
        while not stop.wait(args.publish_every_s):
            version += 1
            perturb_lora_b(peft_model, args.step_size)
            step = (version - 1) * 500
            publisher.publish(peft_model, version=version, step=step, loss=1.0 / version)
            logger.info(f"Published mock adapter v{version} (fake step {step})")
    except KeyboardInterrupt:
        logger.info("Stopping mock trainer")
    finally:
        publisher.stop()


if __name__ == "__main__":
    main()

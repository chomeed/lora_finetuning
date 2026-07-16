"""Local-mode LoRA training entrypoint -- console script ``ws-lora-finetuning``.

Same shared trainer as the remote leg (``common.trainer.train``), but run on the
workstation itself, next to the policy server. "Local mode" just means the
AdapterService and the policy server are on the same box, so bind the service on
loopback and point ``--adapter_addr=127.0.0.1:8090`` at it -- no adapter bytes
ever leave the machine.

    ws-lora-finetuning \
        --pretrained_path=outputs/train/<run>/pretrained_model \
        --dataset_repo_id=chomeed/<dataset> --dataset_root=/data/lerobot/<dataset> \
        --serve_host=127.0.0.1 --serve_port=8090 --publish-freq 500

Both ``--publish-freq`` and ``--publish_freq`` are accepted.
"""

from ..common._cli import normalize_hyphen_flags
from ..common.trainer import train


def main():
    normalize_hyphen_flags()
    train()


if __name__ == "__main__":
    main()

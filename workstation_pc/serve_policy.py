"""Adapter-hot-reloading policy server, workstation leg -- runs the shared
``common.policy_server.serve``. This is the server the robot client connects to;
in local mode it pulls adapters from a trainer on the same box over loopback.

Shortcut (installed by `pip install -e .`): ``ws-serve-policy``.

    ws-serve-policy \
        --host=0.0.0.0 --port=8080 --fps=30 \
        --adapter_addr=127.0.0.1:8090 --reload_on=chunk

Equivalent: ``python -m lora_finetuning.workstation_pc.serve_policy``. Omit
``--adapter_addr`` to run as a plain PolicyServer with no hot-reload.
"""

from ..common._cli import normalize_hyphen_flags
from ..common.policy_server import serve


def main():
    normalize_hyphen_flags()
    serve()


if __name__ == "__main__":
    main()

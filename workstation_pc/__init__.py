"""workstation_pc role (rllab4): the box next to the robot that serves the
policy, ingests demos, and can also train the adapter locally.

The same trainer/policy-server entrypoints also run on a remote GPU box -- just
bind ``--serve_host=0.0.0.0`` there instead of loopback.

Entrypoints:
    realtime_converter.py  `ws-real-time-converter` -- HDF5 -> LeRobot ingest
    train.py               `ws-lora-finetuning` -- LoRA trainer (local or remote)
    serve_policy.py        `ws-serve-policy` -- adapter-hot-reloading policy server
"""

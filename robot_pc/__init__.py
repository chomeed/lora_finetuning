"""robot_pc role (the Orin): pushes finished demo episodes to the workstation's
ingest queue as they are recorded.

Entrypoint:
    demo_sender.py  `robot-demo-sending` -- rsync-over-SSH push into the
                    workstation's tmp_demo ingest dir (the converter's queue)

Deliberately stdlib-only (subprocess rsync/ssh) so it runs in the robot's ROS
env without pulling torch/lerobot/draccus.
"""

"""Observation/action schema constants for the bimanual rig.

The full-rig key layout must stay in lockstep with
``orin_demo_collection.hdf5_writer.STATE_DIMS / ACTION_DIMS`` (and therefore
with the layout of the LeRobot dataset the policy was trained on).

Every inference schema is a :class:`ModeSchema` entry in :data:`MODE_SCHEMAS`,
keyed by the ``--mode`` CLI choice. A schema declares

  * ``state_keys`` / ``action_keys`` — which channels of the full 41-D state /
    19-D action are exposed to the policy,
  * ``publish_*`` — which channel groups ``send_action`` actually drives
    (everything else holds at the pose set during ``reset()``),
  * ``*_arm_unused`` — whether an entire arm+gripper is never read by the
    schema, so RobotEnv can relax its sensor-warmup requirements for that side,
  * ``image_names`` — the image features declared to / sent to the policy
    server.

All modes use the head + both wrist cameras; the external ``third`` camera is
no longer in use.

The realtime converter (workstation_pc) reuses this registry to *project* a
full-rig recording down to the chosen mode's channels before writing the
LeRobot dataset, so the on-disk dataset matches exactly what the policy was
trained on. Projection is by index into the full ``STATE_KEYS`` / ``ACTION_KEYS``
(see :func:`projection_indices`), which is why every mode's keys must be a
subset of the full schema.
"""

from __future__ import annotations

from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Full bimanual schema — 41-D state / 19-D action
# -----------------------------------------------------------------------------

STATE_KEYS: tuple[str, ...] = (
    *[f'arm_left_{i}'  for i in range(7)],
    *[f'arm_right_{i}' for i in range(7)],
    *[f'head_{i}'      for i in range(2)],
    'lift_0',
    *[f'gripper_left_{i}'  for i in range(12)],
    *[f'gripper_right_{i}' for i in range(12)],
)

ACTION_KEYS: tuple[str, ...] = (
    *[f'arm_left_{i}'  for i in range(7)],
    *[f'arm_right_{i}' for i in range(7)],
    *[f'head_{i}'      for i in range(2)],
    'lift_0',
    'gripper_left',
    'gripper_right',
)

# Binary (thresholded) action channels; everything else is continuous.
GRIPPER_ACTION_KEYS: tuple[str, ...] = ('gripper_left', 'gripper_right')

# Reduced-gripper finger-joint subsets. The 12-DoF gripper state is the raw
# message order gl.joint[:12] (snapshot.py), so gripper_*_{i} maps positionally
# to the named joints: i=0..3 -> j_1_1..j_1_4, i=4..7 -> j_2_1..j_2_4,
# i=8..11 -> j_3_1..j_3_4.
_GRIPPER_J1_J3: tuple[int, ...] = (0, 1, 2, 3, 8, 9, 10, 11)   # j_1_* + j_3_*
_GRIPPER_J1_J2: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)     # j_1_* + j_2_*

# Arm-only: drop head_0, head_1, lift_0 from both vectors. Matches the schema
# produced by convert_to_lerobot_arm_only.py — 38-D state, 16-D action.
_ARM_ONLY_DROP: frozenset[str] = frozenset({'head_0', 'head_1', 'lift_0'})

STATE_KEYS_ARM_ONLY: tuple[str, ...] = tuple(
    k for k in STATE_KEYS if k not in _ARM_ONLY_DROP)
ACTION_KEYS_ARM_ONLY: tuple[str, ...] = tuple(
    k for k in ACTION_KEYS if k not in _ARM_ONLY_DROP)

# Left-arm-only: the single-arm schema produced by the chomeed "left_arm_only"
# conversion. Keeps ONLY the left arm + the full 12-DoF left gripper in the
# state, and arm_left + a scalar gripper in the action.
STATE_KEYS_LEFT_ONLY: tuple[str, ...] = (
    *[f'arm_left_{i}'     for i in range(7)],
    *[f'gripper_left_{i}' for i in range(12)],
)
ACTION_KEYS_LEFT_ONLY: tuple[str, ...] = (
    *[f'arm_left_{i}' for i in range(7)],
    'gripper_left',
)

# Handover family: left-arm action + BOTH grippers (one hands the object off
# to the other). The observations differ per variant.
ACTION_KEYS_HANDOVER: tuple[str, ...] = (
    *[f'arm_left_{i}' for i in range(7)],
    'gripper_left',
    'gripper_right',
)
STATE_KEYS_HANDOVER_31: tuple[str, ...] = (
    *[f'arm_left_{i}'      for i in range(7)],
    *[f'gripper_left_{i}'  for i in range(12)],
    *[f'gripper_right_{i}' for i in range(12)],
)
STATE_KEYS_HANDOVER_23: tuple[str, ...] = (
    *[f'arm_left_{i}'      for i in range(7)],
    *[f'gripper_left_{i}'  for i in _GRIPPER_J1_J3],
    *[f'gripper_right_{i}' for i in _GRIPPER_J1_J2],
)

# Insertion-15: single-left-arm schema like left_only, but the left gripper
# observation keeps only j_1_*+j_3_* (8 of 12 DoF; j_2_* dropped).
STATE_KEYS_INSERTION_15: tuple[str, ...] = (
    *[f'arm_left_{i}'     for i in range(7)],
    *[f'gripper_left_{i}' for i in _GRIPPER_J1_J3],
)

# Cable-insertion-first: the RIGHT-arm mirror of insertion_15 — right arm +
# the right gripper's j_1_*+j_2_* finger joints; action is right arm + a
# scalar right gripper.
STATE_KEYS_CABLE_INSERTION_FIRST: tuple[str, ...] = (
    *[f'arm_right_{i}'     for i in range(7)],
    *[f'gripper_right_{i}' for i in _GRIPPER_J1_J2],
)
ACTION_KEYS_CABLE_INSERTION_FIRST: tuple[str, ...] = (
    *[f'arm_right_{i}' for i in range(7)],
    'gripper_right',
)


# -----------------------------------------------------------------------------
# Mode schema registry
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeSchema:
    """Everything mode-dependent about one observation/action schema."""

    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    # Which channel groups send_action publishes. Arms/head/lift not published
    # hold at the pose set during reset(); grippers not published keep their
    # last commanded state. The right arm and right gripper are gated
    # separately because the handover modes drive the right GRIPPER but hold
    # the right ARM.
    publish_left_arm:      bool = True
    publish_right_arm:     bool = True
    publish_left_gripper:  bool = True
    publish_right_gripper: bool = True
    publish_head_lift:     bool = True
    # Whether an entire arm+gripper is never READ by the schema — its gripper
    # sensor / wrist cam / grasp signal don't need to be live to start
    # inference, so RobotEnv/SnapshotBuilder's warmup is relaxed for that
    # side. The handover modes hold the right ARM but still read+drive the
    # right GRIPPER, so they do NOT count as unused here.
    right_arm_unused: bool = False
    left_arm_unused:  bool = False
    # Image features declared to / sent to the policy server. The policy
    # server KeyErrors on any client-declared image key its checkpoint doesn't
    # recognize, so a schema must declare exactly the features the checkpoint
    # was trained with (see 'insertion_15_third').
    image_names: tuple[str, ...] = ('head', 'left_wrist', 'right_wrist')

    @property
    def cont_action_keys(self) -> tuple[str, ...]:
        """Continuous action channels (arms[, head, lift]); grippers are
        binary (hysteresis-thresholded) and excluded."""
        return tuple(k for k in self.action_keys if k not in GRIPPER_ACTION_KEYS)

    @property
    def has_both_grippers(self) -> bool:
        """Whether the ACTION drives both gripper channels (required by
        --swap-grippers)."""
        return frozenset(GRIPPER_ACTION_KEYS) <= frozenset(self.action_keys)


MODE_SCHEMAS: dict[str, ModeSchema] = {
    # 41-D state / 19-D action: both arms, head, lift, both grippers.
    'full': ModeSchema(
        state_keys=STATE_KEYS,
        action_keys=ACTION_KEYS,
    ),
    # 38-D state / 16-D action: both arms + both grippers; head/lift held.
    # Matches the schema produced by convert_to_lerobot_arm_only.py.
    'arm_only': ModeSchema(
        state_keys=STATE_KEYS_ARM_ONLY,
        action_keys=ACTION_KEYS_ARM_ONLY,
        publish_head_lift=False,
    ),
    # 19-D state / 8-D action: left arm + full 12-DoF left gripper only;
    # right arm, right gripper, head, and lift held.
    'left_only': ModeSchema(
        state_keys=STATE_KEYS_LEFT_ONLY,
        action_keys=ACTION_KEYS_LEFT_ONLY,
        publish_right_arm=False,
        publish_right_gripper=False,
        publish_head_lift=False,
        right_arm_unused=True,
    ),
    # 38-D state / 9-D action: bimanual obs (same state as arm_only), but the
    # action drives the left arm + BOTH grippers; right arm/head/lift held.
    'handover': ModeSchema(
        state_keys=STATE_KEYS_ARM_ONLY,
        action_keys=ACTION_KEYS_HANDOVER,
        publish_right_arm=False,
        publish_head_lift=False,
    ),
    # 31-D state / 9-D action: like handover, but the right ARM is also
    # dropped from the observation (left arm + both 12-DoF grippers).
    'handover_31': ModeSchema(
        state_keys=STATE_KEYS_HANDOVER_31,
        action_keys=ACTION_KEYS_HANDOVER,
        publish_right_arm=False,
        publish_head_lift=False,
    ),
    # 23-D state / 9-D action: handover action with a reduced observation —
    # right arm dropped, left gripper j_1_*+j_3_*, right gripper j_1_*+j_2_*.
    'handover_23': ModeSchema(
        state_keys=STATE_KEYS_HANDOVER_23,
        action_keys=ACTION_KEYS_HANDOVER,
        publish_right_arm=False,
        publish_head_lift=False,
    ),
    # 15-D state / 8-D action: left arm + reduced left gripper (j_1_*+j_3_*);
    # right arm/gripper, head, and lift held.
    'insertion_15': ModeSchema(
        state_keys=STATE_KEYS_INSERTION_15,
        action_keys=ACTION_KEYS_LEFT_ONLY,
        publish_right_arm=False,
        publish_right_gripper=False,
        publish_head_lift=False,
        right_arm_unused=True,
    ),
    # 15-D state / 8-D action: RIGHT-arm mirror of insertion_15 — right arm +
    # reduced right gripper (j_1_*+j_2_*); left arm/gripper, head, lift held.
    'cable_insertion_first': ModeSchema(
        state_keys=STATE_KEYS_CABLE_INSERTION_FIRST,
        action_keys=ACTION_KEYS_CABLE_INSERTION_FIRST,
        publish_left_arm=False,
        publish_left_gripper=False,
        publish_right_arm=True,
        publish_head_lift=False,
        left_arm_unused=True,
    ),
}

# Deprecated --mode spellings accepted by the CLI.
MODE_ALIASES: dict[str, str] = {'insertion': 'left_only'}

# Canonical --mode choices (registry keys + accepted aliases), in help order.
MODE_CHOICES: tuple[str, ...] = (
    'full', 'arm_only', 'left_only', 'insertion',
    'handover', 'handover_31', 'handover_23', 'insertion_15',
    'cable_insertion_first',
)
assert frozenset(MODE_CHOICES) == frozenset(MODE_SCHEMAS) | frozenset(MODE_ALIASES)


def resolve_mode(mode: str) -> str:
    """Collapse deprecated aliases to their canonical MODE_SCHEMAS key."""
    return MODE_ALIASES.get(mode, mode)


def get_schema(mode: str) -> ModeSchema:
    """The ModeSchema for a --mode choice (alias-aware)."""
    return MODE_SCHEMAS[resolve_mode(mode)]


def projection_indices(full_keys: tuple[str, ...], subset_keys: tuple[str, ...]) -> list[int]:
    """Column indices that select ``subset_keys`` out of a vector laid out in
    ``full_keys`` order. Used to project a full-rig recording down to a mode.

    Raises KeyError if any subset key is not in the full schema (which the
    per-mode subset assertions below already forbid, but a recording with an
    unexpected layout could still trip)."""
    pos = {k: i for i, k in enumerate(full_keys)}
    return [pos[k] for k in subset_keys]


def _check(mode: str, n_state: int, n_action: int, n_cont: int) -> None:
    s = MODE_SCHEMAS[mode]
    assert len(s.state_keys) == n_state, mode
    assert len(s.action_keys) == n_action, mode
    assert len(s.cont_action_keys) == n_cont, mode
    # Every schema must be a subset of the full one, so the robot can project
    # the full 41-D state down by index lookup (_flatten_built_obs).
    assert frozenset(s.state_keys) <= frozenset(STATE_KEYS), mode
    assert frozenset(s.action_keys) <= frozenset(ACTION_KEYS), mode


_check('full',                  41, 19, 17)
_check('arm_only',              38, 16, 14)
_check('left_only',             19,  8,  7)
_check('handover',              38,  9,  7)
_check('handover_31',           31,  9,  7)
_check('handover_23',           23,  9,  7)
_check('insertion_15',          15,  8,  7)
_check('cable_insertion_first', 15,  8,  7)

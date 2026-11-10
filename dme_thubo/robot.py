"""MJLab entity configuration backed by the checked-in THUBO MJCF."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .model_validation import validate

ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = ROOT / "mjcf" / "THUBO.xml"
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_yaw", "right_hip_roll",
    "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
)
ACTUATOR_NAMES = tuple(f"{name}_ctrl" for name in JOINT_NAMES)


def get_spec() -> mujoco.MjSpec:
    """Load the model and remove XML actuators before MJLab adds position actuators."""
    validate(MJCF_PATH, ROOT / "sts3250_official.json")
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    # The XML actuator section is useful for standalone MuJoCo inspection.  MJLab
    # owns the controller implementation and adds one position actuator per joint.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    # The scene owns the ground plane; avoid attaching a second floor with the robot.
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
    return spec


def get_robot_cfg() -> EntityCfg:
    """Return a fresh floating-base 12-DoF robot configuration."""
    actuator = BuiltinPositionActuatorCfg(
        target_names_expr=(r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee_pitch|ankle_pitch|ankle_roll)$",),
        stiffness=35.0,
        damping=1.3464038511725651,
        effort_limit=8.716130441407099,
        armature=0.03999977737144798,
        frictionloss=0.19999504581400715,
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.01525174942573398, -0.005965198239808444, 0.39),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "left_hip_yaw": 0.0, "left_hip_roll": 0.01800413,
                "left_hip_pitch": -0.34048056, "left_knee_pitch": 0.84851125,
                "left_ankle_pitch": -0.50803069, "left_ankle_roll": -0.01800413,
                "right_hip_yaw": 0.0, "right_hip_roll": 0.01803552,
                "right_hip_pitch": -0.35369690, "right_knee_pitch": 0.87600948,
                "right_ankle_pitch": -0.52231259, "right_ankle_roll": -0.01803552,
            },
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(actuator,),
            soft_joint_pos_limit_factor=0.95,
        ),
    )


def add_ground(spec: mujoco.MjSpec) -> None:
    """Add one MuJoCo ground plane to the MJLab scene."""
    geom = spec.worldbody.add_geom()
    geom.name = "ground"
    geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    geom.size = [3.0, 3.0, 0.1]
    geom.contype = 1
    geom.conaffinity = 1
    geom.friction = [0.8, 0.01, 0.002]

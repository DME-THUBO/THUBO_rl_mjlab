"""THUBO entity constants in the same role as Unitree's robot constants."""

from dme_thubo.robot import ACTUATOR_NAMES, JOINT_NAMES, MJCF_PATH, get_robot_cfg

THUBO_XML = MJCF_PATH
THUBO_JOINT_NAMES = JOINT_NAMES
THUBO_ACTUATOR_NAMES = ACTUATOR_NAMES


def get_thubo_robot_cfg():
    """Return a fresh MJLab EntityCfg for the 12-DoF THUBO model."""
    return get_robot_cfg()

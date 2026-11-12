"""Embodiment adapters; Pi assets retain their upstream provenance and limits."""

from pathlib import Path
import json
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from dme_thubo.robot import get_robot_cfg

ROOT = Path(__file__).resolve().parents[1]


def robot_cfg(name):
    if name == "thubo":
        cfg = get_robot_cfg()
        return cfg, dict(
            height=0.39,
            base="pelvis",
            feet=("left_ankle_roll", "right_ankle_roll"),
            mass=2.946841787,
            reset_height=0.12,
        )
    if name != "pi":
        raise ValueError(name)
    directory = ROOT / "src/assets/robots/pi"
    missing = [
        str(directory / name)
        for name in ("pi.xml", "pi_12dof_release_v1.urdf", "provenance.json")
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing local Pi assets: "
            + ", ".join(missing)
            + ". Generate with: python scripts/host/convert_pi.py /path/to/HoST/legged_gym/resources/robots/pi_12dof"
        )
    path = directory / "pi.xml"

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(path))
        for a in list(spec.actuators):
            spec.delete(a)
        for key in list(spec.keys):
            spec.delete(key)
        return spec

    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f".*{part}_joint",),
            stiffness=kp,
            damping=0.2,
            effort_limit=20.0,
            armature=0.01,
        )
        for part, kp in [
            ("hip_pitch", 30),
            ("hip_roll", 15),
            ("thigh", 15),
            ("calf", 30),
            ("ankle_pitch", 12),
            ("ankle_roll", 5),
        ]
    )
    metadata = json.loads((path.parent / "provenance.json").read_text())
    assert set(metadata["effort_limits"].values()) == {20.0}
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0.351), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=0.9
        ),
    ), dict(
        height=0.351,
        base="base_link",
        feet=("l_ankle_roll_link", "r_ankle_roll_link"),
        mass=metadata["mass"],
        reset_height=0.12,
    )


def pi_assets_available():
    directory = ROOT / "src/assets/robots/pi"
    return all(
        (directory / name).is_file()
        for name in ("pi.xml", "pi_12dof_release_v1.urdf", "provenance.json")
    )

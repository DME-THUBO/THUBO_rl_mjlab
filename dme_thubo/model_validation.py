"""Structural and physical-property checks for the source MuJoCo model."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco

JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_yaw", "right_hip_roll",
    "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
)


def validate(model_path: Path, actuator_path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    root_height = float(data.xpos[root_body, 2])
    mesh_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) for i in range(model.nmesh)]
    metadata = json.loads(actuator_path.read_text())
    checks = {
        "free_base": model.nq == 7 + len(JOINTS),
        "joint_count": sum(name in joint_names for name in JOINTS) == 12,
        "actuator_count": model.nu == 12,
        "root_height_range": 0.30 <= root_height <= 0.50,
        "positive_mass": float(model.body_mass.sum()) > 0.0,
        "mass_range_kg": 2.0 <= float(model.body_mass.sum()) <= 4.0,
        "all_meshes_present": model.nmesh >= 13 and all(name for name in mesh_names),
        "actuator_config_type": metadata.get("actuator_type") == "feetech-sts3250",
    }
    report = {
        "model": str(model_path),
        "mass_kg": float(model.body_mass.sum()),
        "root_height_m": root_height,
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "joint_names": [name for name in joint_names if name],
        "actuator_names": actuator_names,
        "mesh_count": model.nmesh,
        "checks": checks,
        "valid": all(checks.values()),
    }
    if not report["valid"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("MJCF validation failed: " + ", ".join(failed))
    return report

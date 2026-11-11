#!/usr/bin/env python3
"""Visualize THUBO collision geometry in a browser-based Viser viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import viser

import mjviser

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mjcf" / "THUBO.xml"

COLLISION_RGBA = (1.0, 0.25, 0.1, 0.55)  # collision geoms: red
VISUAL_RGBA = (0.55, 0.65, 0.75, 0.30)  # visual meshes: faint blue

PD_KP = 20.0
PD_KD = 1.0


def paint_geoms(model: mujoco.MjModel) -> None:
    """Recolor geoms so collision shapes stand out from visual meshes."""
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name == "floor":
            continue
        if model.geom_contype[i] or model.geom_conaffinity[i]:
            model.geom_rgba[i] = COLLISION_RGBA
        else:
            model.geom_rgba[i] = VISUAL_RGBA


def make_step_fn(model: mujoco.MjModel, home_qpos: np.ndarray):
    """Hold the home pose with gravity compensation plus joint PD."""
    kp = PD_KP
    kd = PD_KD
    ctrl_low = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()
    target = home_qpos[7:].copy()

    def step_fn(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        torque = data.qfrc_bias[6:].copy()
        torque += kp * (target - data.qpos[7:])
        torque -= kd * data.qvel[6:]
        data.ctrl[:] = np.clip(torque, ctrl_low, ctrl_high)
        mujoco.mj_step(model, data)

    return step_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--step",
        action="store_true",
        help="run physics holding the home pose instead of freezing it",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    paint_geoms(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    step_fn = make_step_fn(model, data.qpos.copy()) if args.step else mujoco.mj_forward
    server = viser.ViserServer(host=args.host, port=args.port)
    viewer = mjviser.Viewer(model, data, step_fn=step_fn, server=server)

    print(f"THUBO collision viewer on http://localhost:{args.port}")
    print("  red   = collision geoms")
    print("  faint = visual meshes (no collision)")
    viewer.run()


if __name__ == "__main__":
    main()

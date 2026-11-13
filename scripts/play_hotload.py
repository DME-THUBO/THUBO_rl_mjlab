#!/usr/bin/env python3
"""Play a local MJLab run in Viser and hotload new checkpoints automatically."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import viser

import src.tasks  # noqa: F401  # registers project tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def checkpoint_entries(checkpoint_dir: Path) -> list[tuple[str, str]]:
    now = time.time()
    entries: list[tuple[int, str, str]] = []
    for checkpoint in checkpoint_dir.glob("model_*.pt"):
        try:
            iteration = int(checkpoint.stem.removeprefix("model_"))
        except ValueError:
            continue
        age = max(0, int(now - checkpoint.stat().st_mtime))
        if age < 2:
            continue
        entries.append((iteration, checkpoint.name, format_time_ago(age)))
    entries.sort()
    return [(name, age) for _, name, age in entries]


def i23_command_gui(term, name, server, get_env_idx, on_change=None, request_action=None):
    """Viewer-only controls; handle the trained zero-width lateral range."""
    with server.gui.add_folder(name.capitalize()):
        enabled = server.gui.add_checkbox("직접 속도 조절", initial_value=True)
        forward = server.gui.add_slider("전진 속도 (m/s)", min=0.0, max=0.16, step=0.01, initial_value=0.10)
        lateral = server.gui.add_slider("옆걸음 (고정)", min=-0.001, max=0.001, step=0.001, initial_value=0.0, disabled=True)
        yaw = server.gui.add_slider("회전 (rad/s)", min=-0.10, max=0.10, step=0.01, initial_value=0.0)
        zero = server.gui.add_button("정지 명령")
        @zero.on_click
        def _(_event):
            forward.value = 0.0
            yaw.value = 0.0
    term._joystick_enabled = enabled
    term._joystick_sliders = [forward, lateral, yaw]
    term._joystick_get_env_idx = get_env_idx


class HotloadViewer(ViserPlayViewer):
    def __init__(self, *args, checkpoint_manager: CheckpointManager,
                 training_status_file=None, viewer_state_dir=None, viewer_url=None, **kwargs):
        super().__init__(*args, checkpoint_manager=checkpoint_manager, **kwargs)
        self._checkpoint_poll_at = 0.0
        self._training_status_file = training_status_file
        self._viewer_state_dir = viewer_state_dir
        self._viewer_url = viewer_url

    def setup(self) -> None:
        # Rough heightfields cover a large world. Keep the robot centered in the
        # browser view instead of aiming at the terrain generator's world center.
        self._i23 = self.env.unwrapped.cfg.scene.entities["robot"].spec_fn.__module__.endswith("robot_i23")
        self.cfg.distance = 1.25 if self._i23 else 4.0
        self.cfg.azimuth = 135.0
        self.cfg.elevation = 25.0
        super().setup()
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        self._training_html = None
        if self._i23:
            with self._server.gui.add_folder("I23 보행 학습 · 자동 갱신", expand_by_default=True):
                self._training_html = self._server.gui.add_html("최신 정책을 불러오는 중…")
        self._scene.camera_tracking_enabled = True
        for client in self._server.get_clients().values():
            client.camera.position = np.array([1.0, -1.2, .8] if self._i23 else [2.8, -2.8, 1.9])
            client.camera.look_at = np.array([0.0, 0.0, 0.28 if self._i23 else .4])

    def _process_actions(self) -> None:
        super()._process_actions()
        now = time.monotonic()
        if now < self._checkpoint_poll_at:
            return
        self._checkpoint_poll_at = now + 2.0
        manager = self._ckpt_mgr
        if manager is None:
            return
        entries = manager.fetch_available()
        if entries and entries[-1][0] != manager.current_name:
            self._actions.append((ViewerAction.FETCH_CHECKPOINT, "latest"))
            super()._process_actions()
        if self._i23:
            output = self._viewer_state_dir
            try:
                status = {}
                if self._training_status_file is not None:
                    status = json.loads(self._training_status_file.read_text(encoding="utf-8"))
                robot = self.env.unwrapped.scene["robot"]
                speed = float(robot.data.root_link_lin_vel_b[0, 0])
                receipt = dict(url=self._viewer_url, loaded_checkpoint=manager.current_name,
                    training_updates=status.get("completed_updates"), training_state=status.get("state"),
                    observed_forward_speed_m_s=speed, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    display="Live simulation using latest saved training policy; not a scripted animation")
                (output / "viewer_status.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
                if self._training_html is not None:
                    progress = (f'{status["completed_updates"]:,} / {status.get("planned_updates", 4000):,}회'
                                if "completed_updates" in status else '저장된 정책 재생')
                    self._training_html.content = (f'<b>보행 정책 실시간 재생</b><br>학습: {progress}'
                        f'<br>화면 정책: {manager.current_name}<br>전진 속도: {speed:.3f} m/s'
                        '<br><small>새 저장본을 자동으로 불러옵니다. 숫자가 바뀌면 새 정책의 움직임입니다.</small>')
            except (OSError, ValueError, KeyError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Registered MJLab task ID")
    parser.add_argument("run_dir", type=Path, help="Training run directory")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--status-file", type=Path, help="Optional training status JSON")
    parser.add_argument("--viewer-state-dir", type=Path,
                        help="Viewer receipts directory (default: RUN_DIR/viewer_state)")
    parser.add_argument(
        "--fallen-env-ids",
        type=int,
        nargs="*",
        default=None,
        help="Pin these viewer env indices to the fallen reset pose (getup tasks)",
    )
    args = parser.parse_args()

    configure_torch_backends()
    import warp
    warp.config.kernel_cache_dir = str(ROOT / "logs/warp_cache")
    run_dir = args.run_dir.resolve()
    viewer_state_dir = (args.viewer_state_dir or run_dir / "viewer_state").resolve()
    if args.task == "THUBO-I23-Human-Walk":
        viewer_state_dir.mkdir(parents=True, exist_ok=True)
    available = checkpoint_entries(run_dir)
    if not available:
        raise RuntimeError(f"No completed model_*.pt checkpoint found in {run_dir}")
    checkpoint_name = available[-1][0]

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    if args.fallen_env_ids:
        fallen_term = env_cfg.events.get("reset_fallen_pose")
        if fallen_term is None:
            print(
                "[WARN] --fallen-env-ids ignored: task has no 'reset_fallen_pose' event",
                flush=True,
            )
        else:
            fallen_term.params["fallen_env_ids"] = tuple(args.fallen_env_ids)
            print(
                f"[INFO] Envs {args.fallen_env_ids} start fallen; the other "
                f"{args.num_envs - len(args.fallen_env_ids)} start standing",
                flush=True,
            )
    agent_cfg = load_rl_cfg(args.task)
    device = args.device
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if args.task == "THUBO-I23-Human-Walk":
        command = env.command_manager.get_term("twist")
        command.create_gui = MethodType(i23_command_gui, command)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)

    def load_checkpoint(name: str):
        path = run_dir / name
        runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
        if args.task == "THUBO-I23-Human-Walk":
            record = dict(checkpoint=name, loaded_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
            with (viewer_state_dir / "viewer_hotload.jsonl").open("a",encoding="utf-8") as out:
                out.write(json.dumps(record)+"\n")
        return runner.get_inference_policy(device=device)

    load_checkpoint(checkpoint_name)
    policy = runner.get_inference_policy(device=device)
    manager = CheckpointManager(
        current_name=checkpoint_name,
        fetch_available=lambda: checkpoint_entries(run_dir),
        load_checkpoint=load_checkpoint,
    )
    server = viser.ViserServer(host=args.host, port=args.port, label="THUBO I23 · 보행 학습" if args.task == "THUBO-I23-Human-Walk" else "THUBO PPO")
    print(f"[INFO] Viser ready at http://127.0.0.1:{args.port}", flush=True)
    print(f"[INFO] Watching {run_dir} for model_*.pt checkpoints", flush=True)
    try:
        HotloadViewer(env, policy, viser_server=server, checkpoint_manager=manager,
                      training_status_file=args.status_file, viewer_state_dir=viewer_state_dir,
                      viewer_url=f"http://{args.host}:{args.port}").run()
    finally:
        env.close()


if __name__ == "__main__":
    main()

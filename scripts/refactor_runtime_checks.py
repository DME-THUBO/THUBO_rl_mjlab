"""CUDA reset/step, phase gating, short RSL-RL training and checkpoint replay."""

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    import warp

    warp.config.kernel_cache_dir = str(ROOT / "logs/warp_cache")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from dme_thubo import mdp
    from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg
    import src.tasks  # noqa: F401

    report = {}
    dimensions = {
        "THUBO-Flat-Stand": 42,
        "THUBO-Flat-Velocity": 47,
        "THUBO-Flat-Velocity-Getup": 48,
        "HoST-GetUp-THUBO": 258,
        "HoST-GetUp-PI": 258,
    }
    for task in dimensions:
        if task not in list_tasks():
            continue
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = 4
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
            obs, _ = env.reset()
            assert obs["actor"].shape == (4, dimensions[task])
            assert obs["critic"].shape == obs["actor"].shape
            for _ in range(8):
                obs, reward, *_ = env.step(torch.zeros(4, 12, device=args.device))
                assert (
                    torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
                )
            if task.endswith("Velocity-Getup"):
                robot = env.scene["robot"]
                for height, expected in (
                    (0.12, True),
                    (0.33, True),
                    (0.39, False),
                    (0.33, False),
                ):
                    pose = robot.data.default_root_state[:, :7].clone()
                    pose[:, :3] += env.scene.env_origins
                    pose[:, 2] = height
                    robot.write_root_link_pose_to_sim(pose)
                    env.sim.forward()
                    assert bool((mdp.getup_phase(env) == expected).all()), height
                env.reset(env_ids=torch.tensor([0, 2], device=args.device))
            if task in ("THUBO-Flat-Stand", "THUBO-Flat-Velocity"):
                agent = dme_thubo_ppo_runner_cfg()
                agent.num_steps_per_env = 8
                agent.save_interval = 1
                run = ROOT / "logs/refactor_validation" / task
                run.mkdir(parents=True, exist_ok=True)
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                runner = MjlabOnPolicyRunner(
                    wrapped, asdict(agent), str(run), args.device
                )
                runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
                checkpoint = run / "model_1.pt"
                assert checkpoint.is_file()
                runner.load(str(checkpoint))
                policy = runner.get_inference_policy(device=args.device)
                obs, _ = env.reset()
                for _ in range(8):
                    with torch.no_grad():
                        action = policy(obs)
                    obs, reward, *_ = env.step(action)
                    assert (
                        torch.isfinite(obs["actor"]).all()
                        and torch.isfinite(reward).all()
                    )
            report[task] = {
                "reset_step": True,
                "actor_dim": dimensions[task],
                "short_train_replay": task
                in ("THUBO-Flat-Stand", "THUBO-Flat-Velocity"),
            }
            print("PASS", task, flush=True)
        finally:
            env.close()
    (ROOT / "logs/refactor_runtime.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

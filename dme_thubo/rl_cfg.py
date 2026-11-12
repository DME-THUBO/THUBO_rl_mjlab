"""PPO configuration for THUBO velocity-tracking tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def dme_thubo_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128),
            activation="elu",
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4,
            schedule="adaptive", gamma=0.99, lam=0.95, entropy_coef=0.005,
        ),
        experiment_name="dme_thubo_flat_velocity",
        num_steps_per_env=24,
        max_iterations=4000,
        save_interval=100,
        logger="tensorboard",
        upload_model=False,
    )

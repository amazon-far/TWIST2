"""RSL-RL PPO runner config for Student policy (rsl_rl >= 4.0 API)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1MimicStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for student policy (asymmetric AC: actor=policy, critic=privileged)."""

    class_name: str = "OnPolicyRunner"

    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    save_interval: int = 500
    experiment_name: str = "g1_23dof_mimic_student"
    empirical_normalization: bool = False
    logger: str = "wandb"

    obs_groups: dict = {
        "actor":  ["policy"],
        "critic": ["critic"],
    }

    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

    # DAgger / distillation
    teacher_ckpt: str = ""


@configclass
class G1MimicStudentFuturePPORunnerCfg(G1MimicStudentPPORunnerCfg):
    """Runner for student_future obs variant (CMP)."""
    experiment_name: str = "g1_23dof_mimic_student_future"

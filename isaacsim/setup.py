"""Package setup for g1_23dof_mimic (Isaac Lab / Isaac Sim 5.1.0)."""

from setuptools import find_packages, setup

setup(
    name="g1_23dof_mimic",
    version="0.1.0",
    description="G1 23-DoF Motion Imitation for Isaac Lab (Isaac Sim 5.1.0)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Isaac Lab and rsl_rl are expected to be installed via the
        # isaaclab conda environment; listed here for documentation only.
        # "isaaclab",
        # "isaaclab_rl",
        # "rsl_rl",
        "wandb",
        "termcolor",
    ],
)

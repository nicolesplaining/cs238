"""PPO training with fixed seeds and a saved, reproducible configuration.

PPO: Schulman et al., 2017, https://arxiv.org/abs/1707.06347.
Implementation: Stable-Baselines3, https://github.com/DLR-RM/stable-baselines3.
"""

from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from .env import BatteryEnv


def versions():
    return {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in
            ("gym-anm", "gymnasium", "stable-baselines3", "torch", "numpy", "scipy")}}


def train(config, output, steps=32768, seed=7):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = replace(config, random_start=True, horizon=96)
    config.save(output / "config.json")
    env = Monitor(BatteryEnv(config), str(output / "episodes"))
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", n_steps=512,
                batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=.995,
                gae_lambda=.95, clip_range=.2, ent_coef=.005,
                policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])}, verbose=0)
    model.set_logger(configure(str(output), ["csv"]))
    started = time.perf_counter()
    model.learn(total_timesteps=steps)
    model.save(output / "ppo")
    metadata = {"seed": seed, "requested_steps": steps, "actual_steps": model.num_timesteps,
                "elapsed_seconds": time.perf_counter()-started, "versions": versions(),
                "config": str(output / "config.json"), "policy": "PPO MLP [64, 64]",
                "evaluation": "deterministic mean action; no test-time training"}
    (output / "training.json").write_text(json.dumps(metadata, indent=2)+"\n")
    env.close()
    return metadata

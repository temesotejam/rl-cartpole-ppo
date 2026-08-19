from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .environment import PhysicsConfig, SensorNoiseConfig, make_cartpole_env
from .evaluation import evaluate_episode, evaluate_policy
from .reporting import create_plots, write_metrics_csv, write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = [
    ("25_percent", 0.25),
    ("50_percent", 0.50),
    ("75_percent", 0.75),
    ("100_percent", 1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the continuous cart-pole environment.")
    parser.add_argument("--preset", choices=["quick", "normal", "long"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def load_config(preset: str) -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / f"{preset}.yaml").read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def version_info() -> dict[str, str]:
    names = ["gymnasium", "stable-baselines3", "torch", "numpy"]
    return {name: importlib.metadata.version(name) for name in names}


def record_stage(
    records: list[dict],
    stage: str,
    progress: float,
    timesteps: int,
    policy,
    evaluation_seeds: list[int],
    video_seed: int,
    videos_dir: Path,
    video_index: int,
    physics: PhysicsConfig,
    sensor_noise: SensorNoiseConfig,
) -> None:
    metrics = evaluate_policy(policy, evaluation_seeds, physics, sensor_noise)
    evaluate_episode(
        policy,
        seed=video_seed,
        physics=physics,
        sensor_noise=sensor_noise,
        video_path=videos_dir / f"{video_index:02d}_{stage}.mp4",
    )
    records.append(
        {
            "stage": stage,
            "progress": progress,
            "timesteps": timesteps,
            **metrics.to_dict(),
        }
    )
    print(
        f"[{stage}] steps={timesteps:,} return={metrics.mean_return:.1f} "
        f"survival={metrics.mean_survival_time_s:.2f}s success={metrics.success_rate * 100:.1f}% "
        f"angle={metrics.rms_angle_deg:.2f}deg cart={metrics.rms_cart_position_m:.3f}m"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.preset)
    set_global_seed(args.seed)

    physics = PhysicsConfig(**config["physics"])
    sensor_noise = SensorNoiseConfig(**config["sensor_noise"])
    ppo = config["ppo"]

    output_dir = args.output_dir.resolve()
    models_dir = output_dir / "models"
    videos_dir = output_dir / "videos"
    plots_dir = output_dir / "plots"
    for directory in [output_dir, models_dir, videos_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    evaluation_seeds = [
        args.seed + 100 + index for index in range(int(config["evaluation_episodes"]))
    ]
    video_seed = args.seed + 999
    records: list[dict] = []

    record_stage(
        records,
        "random",
        0.0,
        0,
        None,
        evaluation_seeds,
        video_seed,
        videos_dir,
        0,
        physics,
        sensor_noise,
    )

    env_factory = partial(make_cartpole_env, physics=physics, sensor_noise=sensor_noise)
    env = make_vec_env(
        env_factory,
        n_envs=int(ppo["n_envs"]),
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        use_sde=bool(ppo["use_sde"]),
        sde_sample_freq=int(ppo["sde_sample_freq"]),
        seed=args.seed,
        device="cpu",
        verbose=1,
    )

    total_timesteps = int(config["total_timesteps"])
    for video_index, (stage, fraction) in enumerate(CHECKPOINTS, start=1):
        target = int(total_timesteps * fraction)
        remaining = target - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, reset_num_timesteps=False, progress_bar=False)
        model.save(models_dir / f"{stage}.zip")
        record_stage(
            records,
            stage,
            fraction,
            int(model.num_timesteps),
            model,
            evaluation_seeds,
            video_seed,
            videos_dir,
            video_index,
            physics,
            sensor_noise,
        )

    env.close()
    write_metrics_csv(records, output_dir / "metrics.csv")
    create_plots(records, plots_dir)
    write_summary(records, output_dir / "summary.md", args.preset, args.seed)

    metadata = {
        "environment": "ContinuousCartPole-v1-custom",
        "algorithm": "PPO",
        "preset": args.preset,
        "seed": args.seed,
        "requested_total_timesteps": total_timesteps,
        "actual_final_timesteps": int(records[-1]["timesteps"]),
        "evaluation_seeds": evaluation_seeds,
        "video_seed": video_seed,
        "physics": physics.to_dict(),
        "sensor_noise": sensor_noise.to_dict(),
        "config": config,
        "versions": version_info(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import PPO

from .environment import PhysicsConfig, SensorNoiseConfig
from .evaluation import evaluate_episode, evaluate_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Cart-Pole PPO model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--preset", choices=["quick", "normal", "long"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((REPO_ROOT / "configs" / f"{args.preset}.yaml").read_text(encoding="utf-8"))
    physics = PhysicsConfig(**config["physics"])
    noise = SensorNoiseConfig(**config["sensor_noise"])
    model = PPO.load(args.model, device="cpu")
    seeds = [args.seed + index for index in range(args.episodes)]
    metrics = evaluate_policy(model, seeds, physics, noise)
    if args.video is not None:
        evaluate_episode(model, args.seed, physics, noise, args.video)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()

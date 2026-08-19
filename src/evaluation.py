from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import imageio.v2 as imageio
import numpy as np

from .environment import PhysicsConfig, SensorNoiseConfig, make_cartpole_env


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


@dataclass
class EpisodeMetrics:
    episode_return: float
    survival_time_s: float
    completed: bool
    rms_angle_deg: float
    rms_cart_position_m: float
    rms_cart_velocity_mps: float
    rms_force_n: float
    max_abs_angle_deg: float


@dataclass
class AggregateMetrics:
    mean_return: float
    std_return: float
    mean_survival_time_s: float
    success_rate: float
    rms_angle_deg: float
    rms_cart_position_m: float
    rms_cart_velocity_mps: float
    rms_force_n: float
    max_abs_angle_deg: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _action(policy: PredictPolicy | None, env, observation: np.ndarray) -> np.ndarray:
    if policy is None:
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    action, _ = policy.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def evaluate_episode(
    policy: PredictPolicy | None,
    seed: int,
    physics: PhysicsConfig,
    sensor_noise: SensorNoiseConfig,
    video_path: Path | None = None,
) -> EpisodeMetrics:
    env = make_cartpole_env(
        physics=physics,
        sensor_noise=sensor_noise,
        render_mode="rgb_array" if video_path is not None else None,
    )
    env.action_space.seed(seed + 10_000)
    observation, _ = env.reset(seed=seed)

    frames: list[np.ndarray] = []
    angles: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    forces: list[float] = []
    episode_return = 0.0
    final_truncated = False

    if video_path is not None:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = _action(policy, env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        x, x_dot, theta, _theta_dot = [float(value) for value in info["true_state"]]
        positions.append(x)
        velocities.append(x_dot)
        angles.append(theta)
        forces.append(float(info["actual_force_n"]))
        final_truncated = truncated

        if video_path is not None:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    env.close()

    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=physics.dt_s**-1, macro_block_size=1)

    angle_array = np.asarray(angles, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    velocity_array = np.asarray(velocities, dtype=np.float64)
    force_array = np.asarray(forces, dtype=np.float64)
    duration = len(angles) * physics.dt_s

    return EpisodeMetrics(
        episode_return=episode_return,
        survival_time_s=float(duration),
        completed=bool(final_truncated),
        rms_angle_deg=float(math.degrees(np.sqrt(np.mean(np.square(angle_array))))),
        rms_cart_position_m=float(np.sqrt(np.mean(np.square(position_array)))),
        rms_cart_velocity_mps=float(np.sqrt(np.mean(np.square(velocity_array)))),
        rms_force_n=float(np.sqrt(np.mean(np.square(force_array)))),
        max_abs_angle_deg=float(math.degrees(np.max(np.abs(angle_array)))),
    )


def evaluate_policy(
    policy: PredictPolicy | None,
    seeds: list[int],
    physics: PhysicsConfig,
    sensor_noise: SensorNoiseConfig,
) -> AggregateMetrics:
    episodes = [
        evaluate_episode(policy, seed, physics, sensor_noise)
        for seed in seeds
    ]
    returns = np.asarray([item.episode_return for item in episodes], dtype=np.float64)
    return AggregateMetrics(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_survival_time_s=float(np.mean([item.survival_time_s for item in episodes])),
        success_rate=float(np.mean([item.completed for item in episodes])),
        rms_angle_deg=float(np.mean([item.rms_angle_deg for item in episodes])),
        rms_cart_position_m=float(np.mean([item.rms_cart_position_m for item in episodes])),
        rms_cart_velocity_mps=float(np.mean([item.rms_cart_velocity_mps for item in episodes])),
        rms_force_n=float(np.mean([item.rms_force_n for item in episodes])),
        max_abs_angle_deg=float(np.mean([item.max_abs_angle_deg for item in episodes])),
    )

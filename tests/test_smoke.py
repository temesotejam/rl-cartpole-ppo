from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import PhysicsConfig, SensorNoiseConfig, make_cartpole_env
from src.evaluation import evaluate_policy


def test_environment_step_is_finite() -> None:
    env = make_cartpole_env(
        physics=PhysicsConfig(),
        sensor_noise=SensorNoiseConfig(enabled=False),
    )
    observation, info = env.reset(seed=123)
    assert observation.shape == (5,)
    assert len(info["true_state"]) == 4

    observation, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert observation.shape == (5,)
    assert np.all(np.isfinite(observation))
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    assert np.isfinite(info["actual_force_n"])
    env.close()


def test_sensor_noise_changes_policy_observation_not_true_state() -> None:
    physics = PhysicsConfig()
    clean = make_cartpole_env(physics, SensorNoiseConfig(enabled=False))
    noisy = make_cartpole_env(physics, SensorNoiseConfig(enabled=True))
    clean_obs, clean_info = clean.reset(seed=77)
    noisy_obs, noisy_info = noisy.reset(seed=77)
    assert np.allclose(clean_info["true_state"], noisy_info["true_state"])
    assert not np.allclose(clean_obs, noisy_obs)
    clean.close()
    noisy.close()


def test_short_ppo_training_and_evaluation() -> None:
    physics = PhysicsConfig(max_episode_s=1.0)
    noise = SensorNoiseConfig(enabled=True)
    env = DummyVecEnv([lambda: make_cartpole_env(physics, noise)])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=0.99,
        learning_rate=3e-4,
        seed=123,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=128)
    metrics = evaluate_policy(model, [456], physics, noise)
    assert np.isfinite(metrics.mean_return)
    assert np.isfinite(metrics.rms_angle_deg)
    assert 0.0 <= metrics.success_rate <= 1.0
    env.close()

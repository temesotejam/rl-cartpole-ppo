from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float = 9.81
    cart_mass_kg: float = 1.0
    pole_mass_kg: float = 0.1
    pole_com_length_m: float = 0.5
    max_force_n: float = 10.0
    cart_viscous_friction_n_per_mps: float = 0.10
    motor_time_constant_s: float = 0.05
    dt_s: float = 0.02
    track_limit_m: float = 2.4
    angle_limit_deg: float = 30.0
    max_episode_s: float = 10.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SensorNoiseConfig:
    enabled: bool = True
    position_noise_std_m: float = 0.001
    position_bias_std_m: float = 0.002
    velocity_noise_std_mps: float = 0.01
    velocity_bias_std_mps: float = 0.01
    angle_noise_std_deg: float = 0.25
    angle_bias_std_deg: float = 1.0
    gyro_noise_std_dps: float = 0.10
    gyro_bias_std_dps: float = 0.30

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


class ContinuousCartPoleEnv(gym.Env):
    """Cart-pole with continuous motor force and simple actuator dynamics.

    State convention:
      x          cart position [m], right positive
      x_dot      cart velocity [m/s]
      theta      pole angle [rad], 0 = upright, right-falling positive
      theta_dot  pole angular velocity [rad/s]

    The action is normalized to [-1, 1] and scaled to +/- max_force_n.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, physics: PhysicsConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.physics = physics or PhysicsConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -1.0, -1.0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(4, dtype=np.float64)
        self.actual_force_n = 0.0
        self.commanded_force_n = 0.0
        self.steps = 0

    @property
    def max_steps(self) -> int:
        return int(round(self.physics.max_episode_s / self.physics.dt_s))

    def _observation(self) -> np.ndarray:
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

    def _info(self) -> dict:
        return {
            "true_state": self.state.astype(float).tolist(),
            "commanded_force_n": float(self.commanded_force_n),
            "actual_force_n": float(self.actual_force_n),
            "time_s": float(self.steps * self.physics.dt_s),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        # Balance task: begin near upright but with enough variation that a random
        # controller normally fails quickly.
        self.state = np.array(
            [
                self.np_random.uniform(-0.25, 0.25),
                self.np_random.uniform(-0.10, 0.10),
                self.np_random.uniform(math.radians(-10.0), math.radians(10.0)),
                self.np_random.uniform(math.radians(-8.0), math.radians(8.0)),
            ],
            dtype=np.float64,
        )
        self.actual_force_n = 0.0
        self.commanded_force_n = 0.0
        self.steps = 0
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        p = self.physics
        action_scalar = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.commanded_force_n = action_scalar * p.max_force_n

        # First-order motor/drive response. This gives a realistic small delay
        # without adding a separate electrical model.
        alpha = 1.0 - math.exp(-p.dt_s / max(p.motor_time_constant_s, 1e-6))
        self.actual_force_n += alpha * (self.commanded_force_n - self.actual_force_n)

        x, x_dot, theta, theta_dot = self.state
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        total_mass = p.cart_mass_kg + p.pole_mass_kg
        polemass_length = p.pole_mass_kg * p.pole_com_length_m

        # Cart viscous friction opposes cart motion.
        net_force = self.actual_force_n - p.cart_viscous_friction_n_per_mps * x_dot
        temp = (net_force + polemass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (
            p.gravity * sin_theta - cos_theta * temp
        ) / (
            p.pole_com_length_m
            * (4.0 / 3.0 - p.pole_mass_kg * cos_theta**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass

        # Semi-implicit Euler is stable enough for the 20 ms control interval.
        x_dot += p.dt_s * x_acc
        theta_dot += p.dt_s * theta_acc
        x += p.dt_s * x_dot
        theta += p.dt_s * theta_dot
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        self.state[:] = [x, x_dot, theta, theta_dot]
        self.steps += 1

        angle_scale = math.radians(15.0)
        cost = (
            0.70 * (theta / angle_scale) ** 2
            + 0.15 * (x / 1.2) ** 2
            + 0.05 * (theta_dot / 3.0) ** 2
            + 0.03 * (x_dot / 2.0) ** 2
            + 0.02 * (self.actual_force_n / p.max_force_n) ** 2
        )
        reward = float(1.0 - cost)

        terminated = bool(
            abs(x) > p.track_limit_m or abs(theta) > math.radians(p.angle_limit_deg)
        )
        truncated = bool(self.steps >= self.max_steps)
        if terminated:
            reward -= 5.0

        return self._observation(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        width, height = 800, 450
        image = Image.new("RGB", (width, height), (248, 249, 251))
        draw = ImageDraw.Draw(image)

        rail_y = 315
        left_px, right_px = 70, width - 70
        draw.line((left_px, rail_y + 30, right_px, rail_y + 30), fill=(90, 96, 105), width=4)
        draw.line((left_px, rail_y + 37, right_px, rail_y + 37), fill=(170, 175, 182), width=2)

        x, _, theta, _ = self.state
        normalized = (x + self.physics.track_limit_m) / (2.0 * self.physics.track_limit_m)
        cart_x = left_px + normalized * (right_px - left_px)
        cart_w, cart_h = 90, 42
        cart_box = (
            int(cart_x - cart_w / 2),
            rail_y - cart_h,
            int(cart_x + cart_w / 2),
            rail_y,
        )
        draw.rounded_rectangle(cart_box, radius=8, fill=(51, 65, 85), outline=(26, 34, 45), width=2)
        draw.ellipse((cart_x - 35, rail_y - 5, cart_x - 17, rail_y + 13), fill=(30, 30, 34))
        draw.ellipse((cart_x + 17, rail_y - 5, cart_x + 35, rail_y + 13), fill=(30, 30, 34))

        pivot_x, pivot_y = cart_x, rail_y - cart_h
        pole_px = 175
        tip_x = pivot_x + pole_px * math.sin(theta)
        tip_y = pivot_y - pole_px * math.cos(theta)
        draw.line((pivot_x, pivot_y, tip_x, tip_y), fill=(205, 73, 62), width=10)
        draw.ellipse((pivot_x - 9, pivot_y - 9, pivot_x + 9, pivot_y + 9), fill=(28, 32, 38))
        draw.ellipse((tip_x - 10, tip_y - 10, tip_x + 10, tip_y + 10), fill=(205, 73, 62))

        draw.text((20, 18), f"x = {x:+.3f} m", fill=(30, 35, 42))
        draw.text((20, 42), f"theta = {math.degrees(theta):+.2f} deg", fill=(30, 35, 42))
        draw.text((20, 66), f"force = {self.actual_force_n:+.2f} N", fill=(30, 35, 42))
        draw.text((20, 90), f"t = {self.steps * self.physics.dt_s:.2f} s", fill=(30, 35, 42))
        return np.asarray(image, dtype=np.uint8)


class ConsumerSensorWrapper(gym.ObservationWrapper):
    """Apply consumer-grade position/velocity/IMU noise to policy observations."""

    def __init__(self, env: gym.Env, noise: SensorNoiseConfig | None = None):
        super().__init__(env)
        self.noise = noise or SensorNoiseConfig()
        self._bias = np.zeros(4, dtype=np.float64)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.noise.enabled:
            self._bias = np.array(
                [
                    self.np_random.normal(0.0, self.noise.position_bias_std_m),
                    self.np_random.normal(0.0, self.noise.velocity_bias_std_mps),
                    math.radians(self.np_random.normal(0.0, self.noise.angle_bias_std_deg)),
                    math.radians(self.np_random.normal(0.0, self.noise.gyro_bias_std_dps)),
                ],
                dtype=np.float64,
            )
        else:
            self._bias[:] = 0.0
        return self.observation(observation), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if not self.noise.enabled:
            return np.asarray(observation, dtype=np.float32)

        x = float(observation[0]) + self._bias[0] + self.np_random.normal(0.0, self.noise.position_noise_std_m)
        x_dot = float(observation[1]) + self._bias[1] + self.np_random.normal(0.0, self.noise.velocity_noise_std_mps)
        theta = math.atan2(float(observation[3]), float(observation[2]))
        theta += self._bias[2] + math.radians(self.np_random.normal(0.0, self.noise.angle_noise_std_deg))
        theta_dot = float(observation[4]) + self._bias[3] + math.radians(
            self.np_random.normal(0.0, self.noise.gyro_noise_std_dps)
        )
        return np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)


def make_cartpole_env(
    physics: PhysicsConfig | None = None,
    sensor_noise: SensorNoiseConfig | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    return ConsumerSensorWrapper(
        ContinuousCartPoleEnv(physics=physics, render_mode=render_mode),
        noise=sensor_noise,
    )

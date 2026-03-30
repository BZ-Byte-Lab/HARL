from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import PatternConfig
from world_state import (
    WeekPatternWorld,
    WorldStep,
    action_to_device_vector,
    device_vector_to_action,
)


class PatternEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: PatternConfig | None = None) -> None:
        super().__init__()
        self.config = config or PatternConfig()
        self.action_space = spaces.Discrete(self.config.action_dim)
        self.observation_space = spaces.Box(
            low=np.zeros(self.config.state_dim, dtype=np.float32),
            high=np.ones(self.config.state_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.world_steps: list[WorldStep] = []
        self.current_step = 0
        self.device_state = [0] * self.config.num_devices
        self.previous_device_state = [0] * self.config.num_devices

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        world = WeekPatternWorld(self.config, seed=seed)
        self.world_steps = world.generate_episode()
        self.current_step = 0
        self.device_state = [0] * self.config.num_devices
        self.previous_device_state = [0] * self.config.num_devices

        observation = self._get_observation()
        info = self._info_for_step(self.world_steps[self.current_step])
        return observation, info

    def step(self, action: int):
        current_world = self.world_steps[self.current_step]
        self.previous_device_state = self.device_state.copy()
        self.device_state = action_to_device_vector(action, self.config.num_devices)

        reward_parts = self._reward_components(current_world)
        reward = (
            self.config.reward_weights.habit_match * reward_parts["habit_match"]
            + self.config.reward_weights.comfort * reward_parts["comfort"]
            + self.config.reward_weights.energy * reward_parts["energy"]
            + self.config.reward_weights.switching * reward_parts["switching"]
        )

        self.current_step += 1
        terminated = self.current_step >= self.config.episode_steps
        truncated = False
        observation = (
            np.zeros(self.config.state_dim, dtype=np.float32)
            if terminated
            else self._get_observation()
        )
        info = self._info_for_step(current_world)
        info.update(reward_parts)
        info["selected_action"] = int(action)

        return observation, float(reward), terminated, truncated, info

    def render(self):
        if not self.world_steps:
            return

        step = self.world_steps[min(self.current_step, len(self.world_steps) - 1)]
        print(
            "day=%d slot=%d home=%d light=%.2f devices=%s ideal=%s"
            % (
                step.day_index,
                step.step_in_day,
                step.is_home,
                step.light_level,
                self.device_state,
                step.ideal_devices,
            )
        )

    def close(self):
        return None

    def _get_observation(self) -> np.ndarray:
        step = self.world_steps[self.current_step]
        observation = np.array(
            [
                step.time_of_day_norm,
                step.weekday_norm,
                float(step.is_home),
                step.light_level,
                *[float(value) for value in self.device_state],
            ],
            dtype=np.float32,
        )
        return observation

    def _reward_components(self, world_step: WorldStep) -> dict[str, float]:
        ideal_devices = world_step.ideal_devices
        matching_devices = sum(
            int(device == ideal)
            for device, ideal in zip(self.device_state, ideal_devices, strict=True)
        )
        habit_match = 2.0 * (matching_devices / self.config.num_devices) - 1.0

        comfort = self._comfort_reward(world_step)
        energy = -sum(
            power * state
            for power, state in zip(self.config.power_profile, self.device_state, strict=True)
        ) / sum(self.config.power_profile)
        switching = -sum(
            abs(current - previous)
            for current, previous in zip(
                self.device_state, self.previous_device_state, strict=True
            )
        ) / self.config.num_devices

        return {
            "habit_match": float(habit_match),
            "comfort": float(np.clip(comfort, -1.0, 1.0)),
            "energy": float(energy),
            "switching": float(switching),
        }

    def _comfort_reward(self, world_step: WorldStep) -> float:
        active_lights = sum(self.device_state[:3])

        if not world_step.is_home:
            return 1.0 if sum(self.device_state) == 0 else -(sum(self.device_state) / 5.0)

        reward = 0.2

        if world_step.light_level < self.config.comfort_dark_threshold:
            reward += 0.8 if active_lights > 0 else -1.2
        elif world_step.light_level > self.config.comfort_bright_threshold and active_lights > 0:
            reward -= 0.25 * active_lights

        if world_step.light_level < 0.45 and self.device_state[3] and active_lights == 0:
            reward -= 0.3

        return reward

    def _info_for_step(self, world_step: WorldStep) -> dict:
        return {
            "day_index": world_step.day_index,
            "step_in_day": world_step.step_in_day,
            "is_home": world_step.is_home,
            "light_level": world_step.light_level,
            "ideal_devices": world_step.ideal_devices,
            "ideal_action": device_vector_to_action(world_step.ideal_devices),
        }


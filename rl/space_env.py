"""Gymnasium environment wrapping the SpaceGame.

Adapted from the snake_ai reference. Connects Stable-Baselines3 to the
exact same game object used by the manual and camera modes.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SpaceEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}
    total_step = 0

    def __init__(self, game, render: bool = False):
        super().__init__()
        self.game = game
        self.render_enabled = render

        # 5 discrete actions: left, right, up, down, idle
        self.action_space = spaces.Discrete(5)
        # observation is normalized 0..1, length 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.episode = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.episode += 1
        self.steps = 0
        obs = np.array(self.game.get_state(), dtype=np.float32)
        return obs, {}

    def step(self, action):
        if action == 0:
            self.game.move_left()
        elif action == 1:
            self.game.move_right()
        elif action == 2:
            self.game.move_up()
        elif action == 3:
            self.game.move_down()
        else:
            self.game.idle()

        self.game.update()

        self.steps += 1
        SpaceEnv.total_step += 1

        obs = np.array(self.game.get_state(), dtype=np.float32)
        reward = self.game.get_reward()
        terminated = self.game.is_done()
        truncated = False  # step-limit handled inside the game as terminated

        if self.render_enabled:
            self.game.overlay_lines = [
                f"Episode: {self.episode}",
                f"Steps: {self.steps}",
                f"Total steps: {SpaceEnv.total_step}",
                f"Last reward: {reward:.2f}",
            ]
            self.game.render()

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.game.close()
        super().close()

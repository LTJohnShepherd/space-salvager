"""Watch the trained PPO agent play Space Salvager by itself.

Run:  python ai_play.py
Loads models/space_model.zip and acts with deterministic=True so the saved
policy is reproducible after the program is closed and reopened.
"""

import pygame
from stable_baselines3 import PPO
from game.space_game import SpaceGame
from rl.space_env import SpaceEnv

MODEL_PATH = "models/space_model"


def main():
    game = SpaceGame(render=True, fps=60)
    env = SpaceEnv(game, render=True)
    game.overlay_lines = ["AI Mode", "ESC = quit", "M = mute"]

    model = PPO.load(MODEL_PATH)
    obs, _ = env.reset()

    running = True
    while running:
        # let the window stay responsive and allow quit / mute
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    game.toggle_mute()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    game.close()


if __name__ == "__main__":
    main()

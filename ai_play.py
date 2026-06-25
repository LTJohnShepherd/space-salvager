"""Watch the trained PPO agent play Space Salvager by itself.

Run:  python ai_play.py
Loads models/space_model.zip and acts with deterministic=True so the saved
policy is reproducible after the program is closed and reopened.
"""

from stable_baselines3 import PPO
from game.space_game import SpaceGame
from rl.space_env import SpaceEnv

MODEL_PATH = "models/space_model"


def main():
    game = SpaceGame(render=True, fps=60)
    env = SpaceEnv(game, render=True)

    model = PPO.load(MODEL_PATH)
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()

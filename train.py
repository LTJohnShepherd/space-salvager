"""Train a PPO agent on the Space Salvager environment and save it.

Run:  python train.py
Set RENDER=False for much faster (headless) training.
"""

from stable_baselines3 import PPO
from game.space_game import SpaceGame
from rl.space_env import SpaceEnv

TOTAL_TIMESTEPS = 500_000   # raise for a stronger agent
RENDER = False              # True to watch training (slower)
MODEL_PATH = "models/space_model"


def main():
    game = SpaceGame(render=RENDER, fps=120)
    env = SpaceEnv(game, render=RENDER)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)

    env.close()
    print(f"Saved model: {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()

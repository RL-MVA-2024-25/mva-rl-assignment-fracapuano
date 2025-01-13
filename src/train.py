import os
from argparse import ArgumentParser
from pathlib import Path
from interface import Agent

from stable_baselines3 import PPO


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

models_path = Path(__file__).parent.parent / "models"
model_name = "ppo/best_model.zip"


class ProjectAgent(Agent):
    def act(self, observation, use_random=False):
        # same transformation for the observation that one does when training
        observation = np.log(np.maximum(observation, 1e-6))
        # prediction using SB3's (deterministic) API
        return self.policy.predict(observation, deterministic=True)[0]

    def save(self, path):
        self.policy.save(path)

    def load(self):
        self.policy = PPO.load(models_path / model_name)
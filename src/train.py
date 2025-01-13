import os
from argparse import ArgumentParser
from pathlib import Path
from interface import Agent
import numpy as np

from stable_baselines3 import PPO


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

models_path = Path(__file__).parent.parent / "models"
model_name = "ppo/best_model.zip"


class ProjectAgent(Agent):
    def __init__(self):
        self.buffer_size = 5  # Match the stack_size from training
        self.obs_buffer = None
        self.policy = None

    def act(self, observation, use_random=False):
        # Transform observation as in training
        observation = np.log(np.maximum(observation, 1e-6))
        
        # Initialize buffer if not yet created
        if self.obs_buffer is None:
            self.obs_buffer = np.tile(observation, (self.buffer_size, 1))
        else:
            # Roll the buffer and update the latest observation
            self.obs_buffer = np.roll(self.obs_buffer, shift=-1, axis=0)
            self.obs_buffer[-1] = observation

        # Stack observations to match training environment
        stacked_obs = self.obs_buffer.flatten()
        
        # Get prediction using SB3's API
        return self.policy.predict(stacked_obs, deterministic=True)[0]

    def save(self, path):
        self.policy.save(path)

    def load(self):
        self.policy = PPO.load(models_path / model_name)
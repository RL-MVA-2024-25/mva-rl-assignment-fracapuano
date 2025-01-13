# Standard library imports
import os
from argparse import ArgumentParser
from pathlib import Path

# Third-party imports
import numpy as np
import torch
import gymnasium as gym
from coolname import generate_slug
from gymnasium.wrappers import (
    FrameStackObservation,
    TimeLimit, 
    TransformObservation, 
    TransformReward
)
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    ignore_wandb = False
except ImportError:
    ignore_wandb = True

# Local imports
from env_hiv_fast import FastHIVPatient
from interface import Agent

# Constants
algos_dict = {
    "ppo": PPO,
    "dqn": DQN
}


def build_vec_env(
        domain_randomization: bool, 
        subprocess: bool=False,
        n_envs: int=1, 
        max_steps: int=200):
    """Handle functions to create - possibly parallel - multiple env instances.
    Envs are wrapped with handy """
    def make_env() -> gym.Env:
        env = FastHIVPatient(
            domain_randomization=domain_randomization
        )
        # numerically stabilize observations
        env = TransformObservation(
            env,
            lambda obs: np.log(np.maximum(obs, 1e-6)),
            env.observation_space,
        )
        # scale down the reward for numerical stability
        env =  TransformReward(
            env,
            lambda r: r / 1e8
        )
        # enforces episode termination at max_steps
        env = TimeLimit(
            env, 
            max_episode_steps=max_steps
        )

        # stacks observation, as presented in paper
        env = FrameStackObservation(
            env, 
            stack_size=5
        )

        env = Monitor(env=env)
        return env
    
    if subprocess:
        envs = SubprocVecEnv([make_env for _ in range(n_envs)])
    else: 
        envs = DummyVecEnv([make_env for _ in range(n_envs)])

    return envs

def compute_score(
        random_env_reward: float, 
        deterministic_env_reward: float
    ) -> int:

    score = 0
    if deterministic_env_reward >= 3432807.680391572:
        score += 1
    if deterministic_env_reward >= 1e8:
        score += 1
    if deterministic_env_reward >= 1e9:
        score += 1
    if deterministic_env_reward >= 1e10:
        score += 1
    if deterministic_env_reward >= 2e10:
        score += 1
    if deterministic_env_reward >= 5e10:
        score += 1
    if random_env_reward >= 1e10:
        score += 1
    if random_env_reward >= 2e10:
        score += 1
    if random_env_reward >= 5e10:
        score += 1
    return score

class EvaluationCallback(BaseCallback):
    def __init__(self, eval_episodes: int=50, experiment_name: str=generate_slug()):
        super().__init__(verbose=1)

        self.best_score = float("-inf")
        self.best_random_reward = float("-inf")
        self.best_deterministic_reward = float("-inf")

        self.eval_episodes = eval_episodes

        models_path = Path(__file__).parent.parent / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        self.models_path = models_path
        self.experiment_name = experiment_name

    def _on_step(self):
        random_env = build_vec_env(domain_randomization=True)
        deterministic_env = build_vec_env(domain_randomization=False)

        random_mean_reward, random_std_reward = evaluate_policy(
            self.model, random_env, n_eval_episodes=self.eval_episodes
        )
        deterministic_mean_reward, deterministic_std_reward = evaluate_policy(
            self.model, deterministic_env, n_eval_episodes=self.eval_episodes
        )
        # multiplying by 1e8 to undo reward normalization from build_vec_env
        score = compute_score(
            random_mean_reward * 1e8, 
            deterministic_mean_reward * 1e8
        )
        if score >= self.best_score or (
            random_mean_reward > self.best_random_reward
            and deterministic_mean_reward > self.best_deterministic_reward
        ):
            
            self.model.save(
                os.path.join(
                    self.models_path,
                    self.experiment_name,
                    "best_model.zip"
                )
            )
            if not ignore_wandb:
                wandb.log(
                {
                    "eval/score": score,
                    "eval/random_mean_reward": random_mean_reward,
                    "eval/random_std_reward": random_std_reward,
                    "eval/deterministic_mean_reward": deterministic_mean_reward,
                    "eval/deterministic_std_reward": deterministic_std_reward,
                }
            )
        
        return True

def agent_training(
    algo: str="ppo",
    timesteps: int=2_000_000,
    num_envs: int=1,
    domain_randomization: bool=False,
    
):
    config = {
        "num_envs": num_envs,
        "algo": algo,
        "timesteps": timesteps,
        "domain_randomization": domain_randomization
    }

    run = wandb.init(
        config=config,
        project="hiv-reinforcement-learning",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    
    env = build_vec_env(
        domain_randomization=domain_randomization, 
        subprocess=True if num_envs > 1 else False, 
        n_envs=num_envs
    )

    evalandlog_callback = EvaluationCallback(
        experiment_name=run.name
    )
    eval_callback = EveryNTimesteps(
        n_steps=int(timesteps / 20),  # every 5% of training
        callback=evalandlog_callback
        )

    model = algos_dict[algo](
        "MlpPolicy",
        env,
        tensorboard_log=f"runs/{run.id}",
        ent_coef=0.2,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64])
        )
    )
    wandb_callback = WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2
            )
    
    callbacks_list = [eval_callback] + [wandb_callback] if not ignore_wandb else []

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callbacks_list
    )

    run.finish()

def main():
    parser = ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                       help="Number of training timesteps")
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo",
                       help="RL algorithm to use (ppo or dqn)")
    parser.add_argument("--domain-randomization", action="store_true",
                       help="Enable domain randomization during training")
    
    args = parser.parse_args()
    agent_training(
        num_envs=4, 
        timesteps=args.timesteps, 
        algo=args.algo,
        domain_randomization=args.domain_randomization
    )

if __name__ == "__main__":
    main()


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
        self.obs_dim = 6  # The dimension of a single observation

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

        # Reshape to match expected shape (5, 6)
        stacked_obs = self.obs_buffer.reshape(self.buffer_size, self.obs_dim)
        
        # Get prediction using SB3's API
        return self.policy.predict(stacked_obs, deterministic=True)[0]

    def save(self, path):
        self.policy.save(path)

    def load(self):
        self.policy = PPO.load(models_path / model_name)
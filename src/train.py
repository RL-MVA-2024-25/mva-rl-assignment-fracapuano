from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


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
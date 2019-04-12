import numpy as np
from agent import Agent
from utils import plot, moving_average
import scipy.stats

# Number of agents to use, and how many episodes to train them for
num_agents = 5
num_episodes = 1000

episode_rewards = np.zeros([num_agents, num_episodes])
# Train each agent
for i in range(num_agents):
    agent = Agent(reward_scale=0.1)
    episode_rewards[i], interactions = agent.train(num_episodes)

# Mean reward and standard error per episode
mean_reward = np.mean(episode_rewards, axis=0)
episode_errors = scipy.stats.sem(episode_rewards, 0)

plot(mean_reward, episode_errors, num_agents)
plot(moving_average(mean_reward), episode_errors, num_agents)

print('Mean reward from last 50 episodes', np.mean(mean_reward[-50:]))
print('Mean reward from last 100 episodes', np.mean(mean_reward[-100:]))

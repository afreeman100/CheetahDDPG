import numpy as np
from agent import Agent
from utils import plot

# Number of agents to use, and how many episodes to train them for
num_agents = 5
num_episodes = 1000

episode_rewards = np.zeros([num_agents, num_episodes])
# Train each agent
for i in range(num_agents):
    agent = Agent()
    episode_rewards[i] = agent.train(num_episodes)

# Mean reward per episode
av_reward = np.mean(episode_rewards, axis=0)
plot(av_reward, 'Return per episode, averaged over ' + str(num_agents) + ' agents')

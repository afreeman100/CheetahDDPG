import numpy as np
from agent import Agent
from utils import plot

# Number of agents to use, and how many episodes to train them for
num_agents = 2
num_episodes = 2000

episode_rewards = np.zeros([num_agents, num_episodes])
# Train each agent
for i in range(num_agents):
    agent = Agent(batch_size=128)
    episode_rewards[i], interactions = agent.train(num_episodes)
    # print(episode_rewards[i, -1], 'return after', interactions, 'total interactions')

# Mean reward per episode
av_reward = np.mean(episode_rewards, axis=0)
print('Mean reward from last 50 episodes', np.mean(av_reward[-50:]))
print('Mean reward from last 100 episodes', np.mean(av_reward[-100:]))

plot(av_reward, 'Return per episode, averaged over ' + str(num_agents) + ' agents')

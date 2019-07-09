import numpy as np
from agent import Agent
from utils import plot, moving_average
import scipy.stats
import pickle


def train_and_draw(num_episodes, num_agents):
    """ Train agents sequentially and plot average return. """
    episode_rewards = np.zeros([num_agents, num_episodes])

    for i in range(num_agents):
        agent = Agent()
        episode_rewards[i], interactions = agent.train(num_episodes)

    # Mean reward and standard error per episode
    mean_reward = np.mean(episode_rewards, axis=0)
    episode_errors = scipy.stats.sem(episode_rewards, 0)

    plot(mean_reward, episode_errors, num_agents)
    plot(moving_average(mean_reward), episode_errors, num_agents)

    print('Mean reward from last 50 episodes', np.mean(mean_reward[-50:]))
    print('Mean reward from last 100 episodes', np.mean(mean_reward[-100:]))


def train_and_save(num_episodes):
    """ Train one agents and save returns to a file. """
    agent = Agent()
    episode_rewards, interactions = agent.train(num_episodes)

    with open('agent1.pkl', 'wb') as f:
        pickle.dump(episode_rewards, f)


def load_and_draw(num_episodes, num_agents):
    """ Load returns for a number of agents and plot the average. """

    episode_rewards = np.zeros([num_agents, num_episodes])

    for i in range(num_agents):
        with open('agent' + str(i + 1) + '.pkl', 'rb') as f:
            arr = pickle.load(f)
            episode_rewards[i] = arr

    mean_reward = np.mean(episode_rewards, axis=0)
    episode_errors = scipy.stats.sem(episode_rewards, 0)

    plot(mean_reward, episode_errors, num_agents)
    plot(moving_average(mean_reward), episode_errors, num_agents)

    print('Mean reward from last 50 episodes', np.mean(mean_reward[-50:]))
    print('Mean reward from last 100 episodes', np.mean(mean_reward[-100:]))


# train_and_save(num_episodes=2000)
load_and_draw(num_agents=5, num_episodes=2000)

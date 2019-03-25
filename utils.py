import matplotlib.pyplot as plt
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)


    def add(self, state, action, reward, done, next_state):
        experience = (state, action, reward, done, next_state)
        self.buffer.append(experience)


    def sample_batch(self, num_samples):
        # Ensure not attempting to sample more experiences than there are in the buffer
        num_samples = min(num_samples, len(self.buffer))

        experiences = random.sample(self.buffer, num_samples)

        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        dones = np.array([exp[3] for exp in experiences])
        new_states = np.array([exp[4] for exp in experiences])

        return states, actions, rewards, dones, new_states


    def size(self):
        return len(self.buffer)


    def clear(self):
        self.buffer = deque(maxlen=self.buffer_size)


class Noise:
    """
    Based on the OpenAI baselines implementation, available at
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    """


    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = np.zeros_like(self.mu)


    def get_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(
            size=self.mu.shape)
        self.x_prev = x
        return x


def plot(episode_rewards, title):
    episodes = np.arange(len(episode_rewards)) + 1

    plt.plot(episodes, episode_rewards, label='DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.title(title)
    plt.legend()
    plt.show()

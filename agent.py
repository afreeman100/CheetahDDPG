import tensorflow as tf
import numpy as np
import gym
from tqdm import tqdm
from utils import ReplayBuffer, Noise
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, actor_a=0.0001, critic_a=0.001, tau=0.001, y=0.99, batch_size=64, replay_size=1000000, reward_scale=1):

        self.env = gym.make('HalfCheetah-v2')

        self.state_dimensions = self.env.observation_space.shape[0]
        self.action_dimensions = self.env.action_space.shape[0]
        self.action_bounds = self.env.action_space.high

        self.actor_a = actor_a
        self.critic_a = critic_a
        self.tau = tau
        self.y = y
        self.batch_size = batch_size
        self.reward_scale = reward_scale

        replay_memory_size = replay_size
        self.replay_memory = ReplayBuffer(replay_memory_size)

        tf.reset_default_graph()
        self.save_directory = './model/model.ckpt'


    def load_and_play(self):

        with tf.Session() as sess:

            actor = ActorNetwork(sess, self.actor_a, self.tau, self.batch_size, self.state_dimensions, self.action_dimensions,
                                 self.action_bounds)
            critic = CriticNetwork(sess, self.critic_a, self.tau, self.state_dimensions, self.action_dimensions)
            actor_noise = Noise(mu=np.zeros(self.action_dimensions))

            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, self.save_directory)

            state = self.env.reset()
            while True:
                self.env.render()
                action = sess.run(actor.scaled_out,
                                  feed_dict={actor.inputs: np.reshape(state, (1, self.state_dimensions))}) + actor_noise.get_noise()
                state, reward, done, _ = self.env.step(action[0])
                if done:
                    break


    def train(self, episodes, render=False, verbose=False):

        with tf.Session() as sess:

            actor = ActorNetwork(sess, self.actor_a, self.tau, self.batch_size, self.state_dimensions, self.action_dimensions,
                                 self.action_bounds)
            critic = CriticNetwork(sess, self.critic_a, self.tau, self.state_dimensions, self.action_dimensions)

            actor_noise = Noise(mu=np.zeros(self.action_dimensions))
            sess.run(tf.global_variables_initializer())

            # Set initial target network parameters values
            sess.run(actor.update_target_network_params)
            sess.run(critic.update_target_network_params)

            episode_rewards = np.zeros(episodes)
            best_score = -1000

            for ep in tqdm(range(episodes)):

                state = self.env.reset()

                while True:
                    # Action chosen by actor + noise
                    action = sess.run(actor.scaled_out,
                                      feed_dict={actor.inputs: np.reshape(state, (1, self.state_dimensions))}) + actor_noise.get_noise()

                    # Render the last 10 episodes
                    if render and ep > episodes - 10:
                        self.env.render()

                    # Execute action and store experience
                    next_state, reward, done, _ = self.env.step(action[0])
                    reward *= self.reward_scale
                    self.replay_memory.add(np.reshape(state, (self.state_dimensions,)),
                                           np.reshape(action, (self.action_dimensions,)),
                                           reward, done,
                                           np.reshape(next_state, (self.state_dimensions,)))

                    # Only begin training when there are sufficient experiences in replay memory
                    if self.replay_memory.size() > self.batch_size:
                        state_batch, action_batch, reward_batch, done_batch, next_state_batch = self.replay_memory.sample_batch(
                            self.batch_size)

                        # Calculate Q-value -> use target actor to get action then target critic for its value
                        next_q = sess.run(critic.target_out, feed_dict={
                            critic.target_inputs: next_state_batch,
                            critic.target_action: sess.run(actor.target_scaled_out, feed_dict={actor.target_inputs: next_state_batch})
                        })

                        q_target = np.zeros_like(next_q)
                        for i in range(self.batch_size):
                            if done_batch[i]:
                                q_target[i] = reward_batch[i]
                            else:
                                q_target[i] = reward_batch[i] + self.y * next_q[i]

                        # Update critic network
                        _ = sess.run(critic.optimize, feed_dict={critic.inputs: state_batch, critic.action: action_batch,
                                                                 critic.predicted_q_value: np.reshape(q_target, (self.batch_size, 1))})

                        # Get gradient from critic and use to train actor
                        a_outs = sess.run(actor.scaled_out, feed_dict={actor.inputs: state_batch})
                        grads = sess.run(critic.critic_gradient, feed_dict={critic.inputs: state_batch, critic.action: a_outs})
                        sess.run(actor.optimize, feed_dict={actor.inputs: state_batch, actor.gradient: grads[0]})

                        # Update target networks
                        sess.run(actor.update_target_network_params)
                        sess.run(critic.update_target_network_params)

                    state = next_state
                    episode_rewards[ep] += reward / self.reward_scale

                    if done:
                        if verbose:
                            print('Reward: ', episode_rewards[ep], ' for episode ', ep)

                        # Save model
                        if episode_rewards[ep] > best_score:
                            best_score = episode_rewards[ep]
                            tf.train.Saver().save(sess, self.save_directory)
                            if verbose:
                                print('Saving')

                        break

        return episode_rewards

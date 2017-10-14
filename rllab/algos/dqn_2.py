from rllab.algos.base import RLAlgorithm
import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense

class Agent():
    def __init__(
            self, 
            num_actions,
            env_name,
            frame_width,
            frame_height,
            state_length,
            gamma,
            exploration_steps,
            initial_epsilon,
            final_epsilon,
            initial_replay_size,
            num_replay_memory,
            batch_size,
            target_update_interval,
            train_interval,
            learning_rate,
            momentum,
            min_grad,
            save_interval,
            no_op_steps,
            load_network,
            save_network_path,
            save_summary_path,
            ):
        self.num_actions = num_actions
        self.epsilon = initial_epsilon
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps
        self.t = 0

        self.env_name=env_name
        self.state_length=state_length
        self.frame_width=frame_width
        self.frame_height=frame_height
        self.gamma=gamma
        self.batch_size=batch_size
        self.num_replay_memory=num_replay_memory
        self.initial_replay_size=initial_replay_size
        self.target_update_interval=target_update_interval
        self.train_interval=train_interval
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.min_grad=min_grad
        self.save_interval=save_interval
        self.load_network=load_network
        self.save_network_path=save_network_path
        self.save_summary_path=save_summary_path

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.train.SummaryWriter(self.save_summary_path, self.sess.graph)

        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)

        self.sess.run(tf.initialize_all_variables())

        # Load network
        if self.load_network:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(self.state_length, self.frame_width, self.frame_height)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, self.state_length, self.frame_width, self.frame_height])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum, epsilon=self.min_grad)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (self.frame_width, self.frame_height)) * 255)
        state = [processed_observation for _ in range(self.state_length)]
        return np.stack(state, axis=0)

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < self.initial_replay_size:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > final_epsilon and self.t >= self.initial_replay_size:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > self.num_replay_memory:
            self.replay_memory.popleft()

        if self.t >= self.initial_replay_size:
            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.saver.save(self.sess, self.save_network_path + '/' + self.env_name, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= self.initial_replay_size:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(self.train_interval))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < self.initial_replay_size:
                mode = 'random'
            elif self.initial_replay_size <= self.t < self.initial_replay_size + exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(self.train_interval)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.scalar_summary(self.env_name + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.t += 1

        return action


def preprocess(observation, last_observation, frame_width, frame_height):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (frame_width, frame_height)) * 255)
    return np.reshape(processed_observation, (1, frame_width, frame_height))


class DQN(RLAlgorithm):
    """
    Deep Q Network
    """

    def __init__(
            self,
            env,
            env_name,
            frame_width,
            frame_height,
            num_episodes,
            state_length,
            gamma,
            exploration_steps,
            initial_epsilon,
            final_epsilon,
            initial_replay_size,
            num_replay_memory,
            batch_size,
            target_update_interval,
            train_interval,
            learning_rate,
            momentum,
            min_grad,
            save_interval,
            no_op_steps,
            load_network,
            save_network_path,
            save_summary_path
            ):
        self.env=env
        self.env_name=env_name
        self.frame_width=frame_width
        self.frame_height=frame_height
        self.num_episodes=num_episodes
        self.state_length=state_length
        self.gamma=gamma
        self.exploration_steps=exploration_steps
        self.initial_epsilon=initial_epsilon
        self.final_epsilon=final_epsilon
        self.initial_replay_size=initial_replay_size
        self.num_replay_memory=num_replay_memory
        self.batch_size=batch_size
        self.target_update_interval=target_update_interval
        self.train_interval=train_interval
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.min_grad=min_grad
        self.save_interval=save_interval
        self.no_op_steps=no_op_steps
        self.load_network=load_network
        self.save_network_path=save_network_path
        self.save_summary_path=save_summary_path

    def train(self):
        agent = Agent(
                 num_actions=self.env.action_space.n,
                 env_name=self.env_name,
                 frame_width=self.frame_width,
                 frame_height=self.frame_height,
                 state_length=self.state_length,
                 gamma=self.gamma,
                 exploration_steps=self.exploration_steps,
                 initial_epsilon=self.initial_epsilon,
                 final_epsilon=self.final_epsilon,
                 initial_replay_size=self.initial_replay_size,
                 num_replay_memory=self.num_replay_memory,
                 batch_size=self.batch_size,
                 target_update_interval=self.target_update_interval,
                 train_interval=self.train_interval,
                 learning_rate=self.learning_rate,
                 momentum=self.momentum,
                 min_grad=self.min_grad,
                 save_interval=self.save_interval,
                 no_op_steps=self.no_op_steps,
                 load_network=self.load_network,
                 save_network_path=self.save_network_path,
                 save_summary_path=self.save_summary_path
                 )

        for _ in range(self.num_episodes):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, self.no_op_steps)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()
                processed_observation = preprocess(observation, last_observation, self.frame_width, self.frame_height)
                state = agent.run(state, action, reward, terminal, processed_observation)


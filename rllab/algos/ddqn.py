import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from rllab.algos.base import RLAlgorithm

class DQNAgent:
    def __init__(
            self,
            state_size,
            action_size,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class DDQN(RLAlgorithm):
    def __init__(
            self,
            env,
            batch_size,
            episodes,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            learning_rate):
        self.env=env
        self.batch_size=batch_size
        self.episodes=episodes
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.action_space.n
        self.done=False
        self.gamma=gamma    # discount rate
        self.epsilon=epsilon  # exploration rate
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.learning_rate=learning_rate

    def train(self):
        agent = DQNAgent(
                self.state_size,
                self.action_size,
                self.gamma,
                self.epsilon,
                self.epsilon_min,
                self.epsilon_decay,
                self.learning_rate)

        for e in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(500):
                self.env.render()
                action = agent.act(state)
                next_state, reward, self.done, _ = self.env.step(action)
                reward = reward if not self.done else -10
                next_state = np.reshape(next_state, [1, self.state_size])
                agent.remember(state, action, reward, next_state, self.done)
                state = next_state
                if self.done:
                    agent.update_target_model()
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, self.episodes, time, agent.epsilon))
                    break
            if len(agent.memory) > self.batch_size:
                agent.replay(self.batch_size)
            # if e % 10 == 0:
            #     agent.save("./save/cartpole-ddqn.h5")

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rllab.algos.base import RLAlgorithm

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

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
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class DQN(RLAlgorithm):
    def __init__(
            self,
            env,
            batch_size,
            episodes):
        self.env=env
        self.batch_size=batch_size
        self.episodes=episodes
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.action_space.n
        self.done=False

    def train(self):
        agent = DQNAgent(self.state_size, self.action_size)

        for e in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(500):
                # env.render()
                action = agent.act(state)
                next_state, reward, self.done, _ = self.env.step(action)
                reward = reward if not self.done else -10
                next_state = np.reshape(next_state, [1, self.state_size])
                agent.remember(state, action, reward, next_state, self.done)
                state = next_state
                if self.done:
                    print("episode: {}/{}, score: {}, e: {:2}".format(e, self.episodes, time, agent.epsilon))
            if len(agent.memory) > self.batch_size:
                agent.replay(self.batch_size)
            # if e % 10 == 0:
            #    agent.save("./save/dqn_3.h5")

        

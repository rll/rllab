import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
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

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output Layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_size, activation='linear'))
        # Create the model based on the information above
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)

        # Predict the reward value based on the given state
        act_values = self.model.predict(state)

        # Pick the action based on the predicted reward
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # Sample minibatch from the memory
        minibatch = random.sample(self.memory, batch_size)

        # Extract informations from each memory
        for state, action, reward, next_state, done in minibatch:

            # if done, make our target reward
            target = reward
            if not done:
                # predict the future discounted reward
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the Neural Net with the state and target_f
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
            episodes,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            learning_rate):
        self.env=env    # represents the game environment
        self.batch_size=batch_size
        self.episodes=episodes    # a number of games we want the agent to play.
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.action_space.n
        self.done=False    # whether the game is ended or not
        self.gamma=gamma    # aka decay or discount rate, to calculate the future discounted reward.
        self.epsilon=epsilon  # aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
        self.epsilon_min=epsilon_min    # we want the agent to explore at least this amount.
        self.epsilon_decay=epsilon_decay    # we want to decrease the number of explorations as it gets good at playing games.
        self.learning_rate=learning_rate    # Determines how much neural net learns in each iteration.

    def train(self):
        # initialize the agent
        agent = DQNAgent(
                self.state_size,
                self.action_size,
                self.gamma,
                self.epsilon,
                self.epsilon_min,
                self.epsilon_decay,
                self.learning_rate)

        # Iterate the game
        for e in range(self.episodes):

            # reset state in the beginning of each game
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for time in range(500):
                # turn this on if you want to render
                # self.env.render()
                # Decide action
                action = agent.act(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, self.done, _ = self.env.step(action)
                reward = reward if not self.done else -10
                next_state = np.reshape(next_state, [1, self.state_size])

                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, self.done)

                # make next_state the new current state for the next frame.
                state = next_state

                # done becomes True when the game ends
                # ex) The agent drops the pole
                if self.done:
                    # print the score and break out of the loop
                    self.env.reset()
                    print("episode: {}/{}, score: {}, e: {:2}".format(e, self.episodes, time, agent.epsilon))

            if len(agent.memory) > self.batch_size:
                # train the agent with the experience of the episode
                agent.replay(self.batch_size)
            # if e % 10 == 0:
            #    agent.save("./save/dqn_3.h5")

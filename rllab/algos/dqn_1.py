from rllab.algos.base import RLAlgorithm
import random
import numpy
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K
from theano import printing
from theano.gradient import disconnected_grad

class Agent:
    def __init__(
            self, 
            state_size=None, 
            number_of_actions=1,
            epsilon=0.1, 
            batch_size=32, 
            discount=0.9, 
            memory=50,
            save_name='basic', 
            save_freq=10):

        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.discount = discount
        self.memory = memory
        self.save_name = save_name
        self.states = []
        self.actions = []
        self.rewards = []
        self.experience = []
        self.i = 1
        self.save_freq = save_freq
        self.build_functions()

    def build_model(self):
        S = Input(shape=self.state_size)
        h = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='relu')(S)
        h = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', activation='relu')(h)
        h = Flatten()(h)
        h = Dense(256, activation='relu')(h)
        V = Dense(self.number_of_actions)(h)
        self.model = Model(S, V)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print("loading from {}.h5".format(self.save_name))
        except:
            print("Training a new model")


    def build_functions(self):
        S = Input(shape=self.state_size)
        NS = Input(shape=self.state_size)
        A = Input(shape=(1,), dtype='int32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')
        self.build_model()
        self.value_fn = K.function([S], self.model(S))

        VS = self.model(S)
        VNS = disconnected_grad(self.model(NS))
        future_value = (1-T) * VNS.max(axis=1, keepdims=True)
        discounted_future_value = self.discount * future_value
        target = R + discounted_future_value
        cost = ((VS[:, A] - target)**2).mean()
        opt = RMSprop(0.0001)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        self.train_fn = K.function([S, NS, A, R, T], cost, updates=updates)

    def new_episode(self):
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.states = self.states[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def end_episode(self):
        pass

    def act(self, state):
        self.states[-1].append(state)
        values = self.value_fn([state[None, :]])
        if numpy.random.random() < self.epsilon:
            action = numpy.random.randint(self.number_of_actions)
        else:
            action = values.argmax()
        self.actions[-1].append(action)
        return action, values

    def observe(self, reward):
        self.rewards[-1].append(reward)
        return self.iterate()

    def iterate(self):
        N = len(self.states)
        S = numpy.zeros((self.batch_size,) + self.state_size)
        NS = numpy.zeros((self.batch_size,) + self.state_size)
        A = numpy.zeros((self.batch_size, 1), dtype=numpy.int32)
        R = numpy.zeros((self.batch_size, 1), dtype=numpy.float32)
        T = numpy.zeros((self.batch_size, 1), dtype=numpy.int32)
        for i in range(self.batch_size):
            episode = random.randint(max(0, N-50), N-1)
            num_frames = len(self.states[episode])
            frame = random.randint(0, num_frames-1)
            S[i] = self.states[episode][frame]
            T[i] = 1 if frame == num_frames - 1 else 0
            if frame < num_frames - 1:
                NS[i] = self.states[episode][frame+1]
            A[i] = self.actions[episode][frame]
            R[i] = self.rewards[episode][frame]
        cost = self.train_fn([S, NS, A, R, T])
        return cost


class DQN(RLAlgorithm):
    """
    Deep Q Network
    """

    def __init__(
            self,
            env,
            num_episodes,
            state_size=None, 
            number_of_actions=1,
            epsilon=0.1, 
            batch_size=32, 
            discount=0.9, 
            memory=50,
            save_name='basic', 
            save_freq=10
        ):
        self.env = env
        self.num_episodes = num_episodes
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.discount = discount
        self.memory = memory
        self.save_name = save_name
        self.save_freq = save_freq

    # @overrides
    def train(self):
        agent = Agent(
                state_size=self.env.observation_space.shape,
                number_of_actions=self.env.action_space.n,
                epsilon=self.epsilon,
                batch_size=self.batch_size,
                discount=self.discount,
                memory=self.memory,
                save_name=self.save_name
                ) 

        for e in range(self.num_episodes):
            observation = self.env.reset()
            done = False
            agent.new_episode()
            total_cost = 0.0
            total_reward = 0.0
            frame = 0
            while not done:
                frame += 1
                #env.render()
                action, values = agent.act(observation)
                #action = env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                total_cost += agent.observe(reward)
                total_reward += reward
            print("total reward", total_reward)
            print("mean cost", total_cost/frame)

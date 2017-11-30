import time
import numpy as np
from random import uniform
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from MountainCar import MountainCar


class Learner(object):

    def __init__(self, start, goal, Xrange, Vrange,
                 num_actions, max_memory, hidden_size,
                 learning_rate, discount_factor, epsilon):
        self.env = MountainCar(start, goal, Xrange, Vrange)
        self.num_actions = num_actions
        self.max_memory = max_memory
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.memory = []
        self.episodes = []

    def build_model(self, input_size=2):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_shape=(input_size, ), activation="relu"))

        if self.hidden_size <= 100:
            model.add(Dense(self.hidden_size, activation="sigmoid"))

        model.add(Dense(self.num_actions, activation="linear"))
        model.compile(Adam(lr=self.learning_rate), "mse")

        return model

    def get_initial_state(self):
        lhand, rhand = self.env.start[0] * 0.8, self.env.start[0] * 1.2
        start = round(uniform(lhand, rhand), 2)

        self.env.state = np.array([start, self.env.start[1]])  # Reset env
        return self.env.observe()

    def remember(self, experience):
        # experience (DoubleQ) -> [[state, action, reward, next_state], game_over]
        # experience (Sarsa)   -> [[state, action, reward], game_over]

        self.memory.append(experience)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    @staticmethod
    def print_time_passed(start_time):
        time_passed = time.time() - start_time
        print("Time passed: %.3f seconds.." % time_passed)

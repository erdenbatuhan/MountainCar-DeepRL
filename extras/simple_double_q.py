import json
import math
import itertools
import numpy as np
from random import random
import matplotlib.pyplot as plt
from MountainCar import MountainCar


def report(episode, step, position, action, dist, best_dist):
    print("Epoch: %d \t| Step: %d \t| Pos: %.3f \t| Act %d \t| Dist %.3f \t| Best Dist %.3f" %
          (episode, step, position, int(action), dist, best_dist))


def get_position(state):
    position = state[0][0]
    return float("%.3f" % position)


class SimpleDoubleQ(object):

    def __init__(self, env, learning_rate=.005, discount_factor=.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.episodes = []

    def prepare_policies(self, *policies, state):
        for policy in policies:
            if policy.get(get_position(state), None) is None:
                policy[get_position(state)] = np.zeros(self.env.num_actions)

    def update_policy(self, Q1, Q2, state, action, reward, next_state):
        self.prepare_policies(Q1, Q2, state=next_state)

        next_action = np.argmax(Q1[get_position(next_state)])
        target = reward + self.discount_factor * Q2[get_position(next_state)][next_action]
        delta = target - Q1[get_position(state)][action]

        Q1[get_position(state)][action] += self.learning_rate * delta

    def train(self, epoch=100):
        start_time = time.time()
        self.episodes = []

        Q1, Q2 = {}, {}
        Q1[get_position([self.env.goal])], Q2[get_position([self.env.goal])] = [100], [100]

        for episode in range(epoch):
            self.env.reset()

            state = self.env.observe()
            best_dist = float(math.inf)

            for step in itertools.count():
                self.prepare_policies(Q1, Q2, state=state)

                action = np.argmax(Q1[get_position(state)] + Q2[get_position(state)])
                next_state, reward, game_over = self.env.act(action)

                if game_over:  # Terminal state
                    report(episode, step, get_position([self.env.goal]), -1, 0., 0.)
                    break

                if random() > .5:
                    self.update_policy(Q1, Q2, state, action, reward, next_state)
                else:
                    self.update_policy(Q2, Q1, state, action, reward, next_state)

                dist = get_position([self.env.goal]) - get_position(state)
                best_dist = min(best_dist, dist)

                report(episode, step, get_position(state), action, dist, best_dist)
                state = next_state

            self.episodes.append(step)

        self.print_time_passed(start_time)

    def plot(self):
        plt.plot(self.episodes)
        plt.xlabel("Episode")
        plt.ylabel("Length of Episode")
        plt.show()

    @staticmethod
    def print_time_passed(start_time):
        print("Time passed:", end=" ")
        time_passed = time.time() - start_time
        time.strftime("%H:%M:%S", time.gmtime(time_passed))


def main():
    start = [-.5, 0.]
    goal = [.45]
    Xrange = [-1.5, .55]
    Vrange = [-2., 2.]

    env = MountainCar(start, goal, Xrange, Vrange)
    simple_double_q = SimpleDoubleQ(env)

    simple_double_q.train()
    simple_double_q.plot()


if __name__ == "__main__":
    main()


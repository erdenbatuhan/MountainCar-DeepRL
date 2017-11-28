import time
import json
import numpy as np
import matplotlib.pyplot as plt
from random import random
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from MountainCar import MountainCar


def report(step, episode, epoch, loss, win_count, position, action):
    print("Step {} Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {} | Pos {:.3f} | Act {}".
          format(step, episode, (epoch - 1), loss, win_count, position, action))


class DDQN(object):

    def __init__(self, env, num_actions=3, max_memory=500):
        self.env = env
        self.num_actions = num_actions
        self.max_memory = max_memory

        self.memory, self.models, self.episodes = [], [], []
        self.build_models()

    def build_models(self, hidden_size=32, input_size=2, learning_rate=.001):
        self.models = []

        for i in range(2):
            model = Sequential()
            model.add(Dense(hidden_size, input_shape=(input_size, ), activation="relu"))
            model.add(Dense(hidden_size, activation="relu"))
            model.add(Dense(self.num_actions, activation="linear"))
            model.compile(Adam(lr=learning_rate), "mse")      

            self.models.append(model)

    def remember(self, states, game_over):
        # states -> [state, action, reward, next_state]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=50, discount_factor=.99):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]

        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state
            targets[i] = model.predict(state)[0]

            if game_over:
                targets[i, action] = reward
            else:
                q = self.models[0].predict(next_state)[0]
                qq = self.models[1].predict(next_state)[0]

                targets[i, action] = reward + discount_factor * qq[np.argmax(q)]

        return inputs, targets

    def reset_weights(self):
        self.models[1].set_weights(self.models[0].get_weights())

    def train(self, epoch=100, epsilon=.1):
        start_time = time.time()
        win_count = 0
        self.episodes = []

        for episode in range(epoch):
            self.env.reset()
            state = self.env.observe()

            loss = 0.
            game_over = False

            step = 0
            while not game_over:
                step += 1
                if episode > 10 and step > 1000:
                    break

                model = self.models[0]

                # get next action
                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, self.num_actions, size=1)[0]
                else:
                    q = model.predict(state)[0]
                    action = np.argmax(q)

                # apply action, get rewards and new state
                next_state, reward, game_over = self.env.act(action)
                if reward == 100:
                    win_count += 1

                self.remember([state, action, reward, next_state], game_over)  # store experience
                inputs, targets = self.get_batch(model)  # adapt model
                loss += model.train_on_batch(inputs, targets)

                if step % 100 == 1 or game_over:
                    report(step, episode, epoch, loss, win_count, state[0, 0], action)
                state = next_state

            self.episodes.append(step)
            self.reset_weights()

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
    ddqn = DDQN(env)

    # If you want to continue training from a previous model, just uncomment the line bellow
    # ddqn.models[0].load_weights("ddqn-model.h5")

    ddqn.train()
    ddqn.plot()

    # Save trained model weights and architecture, this will be used by the visualization code
    ddqn.models[0].save_weights("ddqn-model.h5", overwrite=True)
    with open("ddqn-model.json", "w") as outfile:
        json.dump(ddqn.models[0].to_json(), outfile)


if __name__ == "__main__":
    main()


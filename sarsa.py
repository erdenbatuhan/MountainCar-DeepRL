import time
import math
import numpy as np
from learner import Learner


class Sarsa(Learner):

    def __init__(self, start, goal, Xrange, Vrange,
                 num_actions=3, max_memory=500, hidden_size=200,
                 learning_rate=.001, discount_factor=.99, epsilon=.2, n=8):
        Learner.__init__(self, start, goal, Xrange, Vrange,
                         num_actions, max_memory, hidden_size,
                         learning_rate, discount_factor, epsilon)
        self.n = n
        self.model = self.build_model()

    def get_next_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions, size=1)[0]
        else:
            action = np.argmax(self.model.predict(state)[0])

        return action

    def get_batch(self, G, batch_size=50):
        len_memory = len(self.memory)
        env_dim = self.memory[0][0][0].shape[1]

        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], self.num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state, action, reward = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state
            targets[i] = self.model.predict(state)[0]

            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = G

        return inputs, targets

    def train(self, epoch, max_episode_length):
        start_time = time.time()
        win_count = 0
        self.episodes = []

        for episode in range(epoch):
            state = self.get_initial_state()
            action = self.get_next_action(state)
            reward = 0
            game_over = False

            # mini_memory -> [S, A, R]
            mini_memory = list()
            mini_memory.append([state, action, reward])

            t = 0
            T = float(math.inf)
            while True:
                t += 1

                if t > max_episode_length and not game_over:
                    break
                if t < T:
                    next_state, reward, game_over = self.env.act(action)
                    next_action = None

                    if game_over:
                        win_count += 1
                        T = t
                    else:
                        next_action = self.get_next_action(next_state)

                    mini_memory.append([next_state, next_action, reward])
                    action = next_action

                t_ = t - self.n
                if t_ >= 0:
                    G = sum([(self.discount_factor ** (i - t_ - 1)) * mini_memory[i][2]
                             for i in range(t_ + 1, min(t_ + self.n, T) + 1)])
                    if t_ + self.n < T:
                        Q = self.model.predict(mini_memory[t_ + self.n][0])[0]
                        G += (self.discount_factor ** self.n) * Q[mini_memory[t_ + self.n][1]]

                    # semi-gradient
                    self.remember([mini_memory[t_], game_over])  # store experience
                    inputs, targets = self.get_batch(G)  # adapt model
                    self.model.train_on_batch(inputs, targets)

                if t % 100 == 1 or game_over:
                    print("n {} | Step {} Epoch {:03d}/{:03d} | Win count {}".
                          format(self.n, t, episode, (epoch - 1), win_count))
                if t_ == T - 1:
                    break

            self.episodes.append(t)

        self.print_time_passed(start_time)

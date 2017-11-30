import time
import numpy as np
from random import random
from learner import Learner


class DoubleQ(Learner):

    def __init__(self, start, goal, Xrange, Vrange,
                 num_actions=3, max_memory=500, hidden_size=200,
                 learning_rate=.001, discount_factor=.99, epsilon=.1):
        Learner.__init__(self, start, goal, Xrange, Vrange,
                         num_actions, max_memory, hidden_size,
                         learning_rate, discount_factor, epsilon)
        self.model1, self.model2 = self.build_model(), self.build_model()

    def get_next_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions, size=1)[0]
        else:
            Q1, Q2 = self.model1.predict(state)[0], self.model2.predict(state)[0]
            action = np.argmax(np.add(Q1, Q2))

        return action

    def get_batch(self, batch_size=50):
        len_memory = len(self.memory)
        env_dim = self.memory[0][0][0].shape[1]

        prob = random()
        model = self.model1 if prob > .5 else self.model2

        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], self.num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state
            targets[i] = model.predict(state)[0]

            Q1 = self.model1.predict(next_state)[0]
            Q2 = self.model2.predict(next_state)[0]

            if game_over:
                targets[i, action] = reward
            else:
                if prob > .5:
                    targets[i, action] = reward + self.discount_factor * Q2[np.argmax(Q1)]
                else:
                    targets[i, action] = reward + self.discount_factor * Q1[np.argmax(Q2)]

        return model, inputs, targets

    def train(self, epoch, max_episode_length):
        start_time = time.time()
        win_count = 0
        self.episodes = []

        for episode in range(epoch):
            state = self.get_initial_state()
            loss = 0.
            game_over = False

            step = 0
            while not game_over:
                step += 1
                if step > max_episode_length:
                    break

                action = self.get_next_action(state)
                next_state, reward, game_over = self.env.act(action)

                if game_over:
                    win_count += 1

                self.remember([[state, action, reward, next_state], game_over])  # store experience
                model, inputs, targets = self.get_batch()  # adapt model
                loss += model.train_on_batch(inputs, targets)

                if step % 100 == 1 or game_over:
                    print("Step {} Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {} | Pos {:.3f} | Act {}".
                          format(step, episode, (epoch - 1), loss, win_count, state[0, 0], (action - 1)))

                state = next_state

            self.episodes.append(step)

        self.print_time_passed(start_time)

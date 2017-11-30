from double_q import DoubleQ
from sarsa import Sarsa
import matplotlib.pyplot as plt


EPOCH = 20
MAX_EPISODE_LENGTH = 10000


def plot(plot_data):
    plt.plot(plot_data)
    plt.xlabel("Episode")
    plt.ylabel("Length of Episode")
    plt.show()


def plot_with_n(plot_data):
    plt.subplot(plot_data[0])

    for i in range(1, len(plot_data)):
        label = "n=" + str(i)
        plt.plot(plot_data[i], label=label)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def run_double_q(start, goal, Xrange, Vrange):
    double_q = DoubleQ(start, goal, Xrange, Vrange)
    double_q.train(epoch=EPOCH, max_episode_length=MAX_EPISODE_LENGTH)

    plot(double_q.episodes)


def run_sarsa(start, goal, Xrange, Vrange, plot_data_pid):
    sarsa_plot_data = list()
    sarsa_plot_data.append(plot_data_pid)

    for i in range(1, 9):
        sarsa = Sarsa(start, goal, Xrange, Vrange, n=i)
        sarsa.train(epoch=EPOCH, max_episode_length=MAX_EPISODE_LENGTH)

        sarsa_plot_data.append(sarsa.episodes)

    plot_with_n(sarsa_plot_data)


def main():
    plot_data_pids = [223]

    start = [-.5, 0.]
    goal = [.45]
    Xrange = [-1.5, .55]
    Vrange = [-2., 2.]

    #run_double_q(start, goal, Xrange, Vrange)
    run_sarsa(start, goal, Xrange, Vrange, plot_data_pids[0])


if __name__ == "__main__":
    main()

from algorithms import run_algorithm
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Statistics import Statistics
from joblib import Parallel, delayed
import itertools


def run_all_algos(deltas, Ks, algos, num_repeat, T):
    for delta in deltas:
        for K in Ks:
            for algo in algos:
                for i in range(num_repeat):
                    statistics = run_algorithm(algo, delta, K, T)
                    save_path = "./results/{}_{}_{}_{}.pickle".format(algo, delta, K, i + 1)
                    with open(save_path, 'wb') as handle:
                        pickle.dump(statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_algo_par(i, deltas, Ks, algos, num_repeat, T):
    list_pos = list(itertools.product(deltas, Ks, algos, list(range(num_repeat))))
    curr_pos = list_pos[i]
    statistics = run_algorithm(curr_pos[2], curr_pos[0], curr_pos[1], T)
    save_path = "./results/{}_{}_{}_{}.pickle".format(curr_pos[2], curr_pos[0], curr_pos[1], int(curr_pos[3]) + 1)
    with open(save_path, 'wb') as handle:
        pickle.dump(statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot_all_graphs(deltas, Ks, algos, num_repeat):
    """
    Plot all the graphs required for this exercise - regret and explorationn index
    :param deltas:
    :param Ks:
    :param algos:
    :param num_repeat:
    :return:
    """
    rows_num = len(deltas)
    cols_num = len(Ks)
    plt.figure("regrets", figsize=(20, 10))
    plt.figure("exploration_index", figsize=(20, 10))
    # plt.figure("means", figsize=(20, 10))
    for al_idx, algo in enumerate(algos):
        for c, K in enumerate(Ks):
            for r, delta in enumerate(deltas):
                all_stats = []
                for rep in range(num_repeat):
                    load_path = "./results/{}_{}_{}_{}.pickle".format(algo, delta, K, rep + 1)
                    with open(load_path, 'rb') as handle:
                        statistics = pickle.load(handle)
                        all_stats.append(statistics)
                plt.figure("regrets")
                plt.subplot(rows_num, cols_num, r*cols_num + c + 1)
                plot_regret(all_stats, delta, al_idx)

                plt.figure("exploration_index")
                plt.subplot(rows_num, cols_num, r * cols_num + c + 1)
                plot_exploration_index(all_stats, al_idx)

                # plt.figure("means")
                # plt.subplot(rows_num, cols_num, r * cols_num + c + 1)
                # plot_means(all_stats, al_idx)

    plt.figure("regrets")
    plt.figlegend(["greedy", "ucb1", "ucbv", "thompson a=b=1"], loc='lower center', ncol=5, labelspacing=0.)
    plt.suptitle("Regrets")
    plt.savefig("./results/regrets.png")
    plt.figure("exploration_index")
    plt.suptitle("Exploration Index")
    plt.figlegend(["greedy", "ucb1", "ucbv", "thompson a=b=1"], loc='lower center', ncol=5, labelspacing=0.)
    plt.savefig("./results/exploration_index.png")
    # plt.figure("means")
    # plt.savefig("./results/means.png")


def plot_regret(all_stats, delta, al_idx):
    """
    Plot the mean regret with std
    :param all_stats: all of the statistics gathered during the current algo run
    :param delta: the current delta
    :param al_idx: the index of the algorithm for deciding the color of the plot
    :return:
    """
    colors = ["red", "blue", "green", "yellow"]
    T = len(all_stats[0].history)
    regrets = np.zeros([len(all_stats), T])
    for idx, statistics in enumerate(all_stats):
        regrets[idx] = statistics.get_regret(delta)
    mean_regrets = np.mean(regrets, axis=0)
    var_regrets = np.var(regrets, axis=0)

    plt.plot(range(T), mean_regrets,
                     color=colors[al_idx])
    plt.fill_between(range(T), mean_regrets - np.sqrt(var_regrets), mean_regrets + np.sqrt(var_regrets),
                 color=colors[al_idx], alpha=0.2)


def plot_exploration_index(all_stats, al_idx):
    """
    Plot the mean exploration_index with std
    :param all_stats: all of the statistics gathered during the current algo run
    :param al_idx: the index of the algorithm for deciding the color of the plot
    :return:
    """
    colors = ["red", "blue", "green", "yellow"]
    T = len(all_stats[0].history)
    eis = np.zeros([len(all_stats), T])
    for idx, statistics in enumerate(all_stats):
        eis[idx] = statistics.explore_index
    mean_regrets = np.mean(eis, axis=0)
    var_regrets = np.var(eis, axis=0)

    plt.plot(range(T), mean_regrets,
                     color=colors[al_idx])
    plt.fill_between(range(T), mean_regrets - np.sqrt(var_regrets), mean_regrets + np.sqrt(var_regrets),
                 color=colors[al_idx], alpha=0.2)


def plot_means(all_stats, al_idx):
    """
    Plot the mean exploration_index with std
    :param all_stats: all of the statistics gathered during the current algo run
    :param al_idx: the index of the algorithm for deciding the color of the plot
    :return:
    """
    colors = ["red", "blue", "green", "yellow"]
    T = len(all_stats[0].history)
    all_means = all_stats[0].all_averages
    all_vars = all_stats[0].all_variances
    all_means = np.asarray([all_means[i][all_stats[0].optimal_arm] for i in range(T)])
    all_vars = np.asarray([all_vars[i][all_stats[0].optimal_arm] for i in range(T)])
    all_means[all_means == np.inf] = 0
    all_vars[all_vars == np.inf] = 0
    plt.plot(range(T), all_means, color=colors[al_idx])
    plt.fill_between(range(T), all_means - np.sqrt(all_vars), all_means + np.sqrt(all_vars),
                color=colors[al_idx], alpha=0.2)


def main():
    """
    The main function
    """
    # deltas = [0.01]
    deltas = [0.1, 0.01]
    # Ks = [100]
    Ks = [2, 10, 100]
    # algos = ["ucbv"]
    # algos = ["greedy", "ucb1", "ucbv", "thompson1"]
    algos = ["thompson2", "thompson3"]

    T = 10 ** 7
    num_repeat = 1

    tot_num_proc = num_repeat*len(algos)*len(Ks)*len(deltas)
    num_par_proc = 10
    Parallel(n_jobs=num_par_proc, verbose=10)(delayed(run_algo_par)(i, deltas, Ks, algos, num_repeat, T) for i in range(tot_num_proc))
    # plot_all_graphs(deltas, Ks, algos, num_repeat)


if __name__ == '__main__':
    main()





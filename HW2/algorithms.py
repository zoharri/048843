import numpy as np
import random
from scipy.stats import bernoulli
import arm_utils
from tqdm import tqdm
from Statistics import Statistics

def run_algorithm(alg_name, delta, K, num_steps):
    """
    Run one of the implemented algorithms
    :param alg_name: which algorithm to run
    :param delta: the parameter for the optimal arm distribution
    :param K: number of arms
    :param num_steps: parameter for the exploration
    :return: returns the statistics after the run
    """
    optimal_arm = random.randint(0, K-1)
    statistics = Statistics(K, optimal_arm)
    print("running algo: {} with delta={}, K={} and T={}".format(alg_name, delta, K, num_steps))
    if alg_name == "greedy":
        c = delta
        d = delta
        statistics = run_dec_eps_greedy(statistics, K, c, d, delta, num_steps)
    elif alg_name == "ucb1":
        statistics = run_ucb1(statistics, num_steps, delta)
    elif alg_name == "thompson1":
        alpha = 1
        beta = alpha
        statistics = run_thompson_sampling(statistics, alpha, beta, K, delta, num_steps)
    elif alg_name == "ucbv":
        b = 1
        c = 1
        statistics = run_ucbv(statistics, c, b, delta, num_steps)
    else:
        print("unknown algorithm name")
        return statistics

    return statistics


def run_dec_eps_greedy(statistics, K, c, d, delta, num_steps):
    """
    The decaying epsilon greedy algorithm
    :param statistics: the statistics of rewards until this time step
    :param K: number of arms
    :param c: parameter for the exploration
    :param d: parameter for the exploration
    :return: returns the statistics after the run
    """
    # for _ in tqdm(range(num_steps)):
    for _ in range(num_steps):
        t = statistics.t+1
        averages = statistics.averages
        eps = np.min((1, c*K/(d**2*t)))
        explore = bernoulli.rvs(eps)
        if explore:  # choose a random arm (EXPLORE)
            arm_idx = random.randint(0, K-1)
        else:  # choose best arm (EXPLOIT - GREEDY)
            arm_idx = np.argmax(averages)

        reward = arm_utils.get_reward(arm_idx == statistics.optimal_arm, delta)
        statistics.update_statistics(arm_idx, reward)
    return statistics


def run_ucb1(statistics, T, delta):
    """
    The UCB1 algorithm
    :param statistics: the statistics of rewards until this time step
    :param T: total number of time steps
    :param delta: the parameter for the optimal arm distribution
    :return: returns the statistics after the run
    """
    num_steps = T
    # for _ in tqdm(range(num_steps)):
    for _ in range(num_steps):
        ucbs = arm_utils.get_ucbs(statistics.averages, statistics.nums, T)
        arm_idx = np.argmax(ucbs)

        reward = arm_utils.get_reward(arm_idx == statistics.optimal_arm, delta)
        statistics.update_statistics(arm_idx, reward)
    return statistics


def run_ucbv(statistics, c, b, delta, num_steps):
    """
    The UCB-V algorithm
    :param statistics: the statistics of rewards until this time step
    :param c: parameter to calculate Bs
    :param b: parameter to calculate Bs
    :param delta: the parameter for the optimal arm distribution
    :param num_steps: number of times to run the algorithm
    :return: returns the statistics after the run
    """
    for _ in tqdm(range(num_steps)):
    # for _ in range(num_steps):
        t = statistics.t + 1
        averages = statistics.averages
        em_vars = statistics.em_vars
        nums = statistics.nums
        Bs = arm_utils.get_Bs(t, averages, em_vars, nums, c, b)

        arm_idx = np.argmax(Bs)
        reward = arm_utils.get_reward(arm_idx == statistics.optimal_arm, delta)
        statistics.update_statistics(arm_idx, reward)
    return statistics

def run_thompson_sampling(statistics, alpha, beta, K, delta, num_steps):
    """
    The Thompson sampling algorithm algorithm with beta distribution as a prior (parameters given as input)
    :param statistics: the statistics of rewards until this time step
    :param alpha: parameter of the prior distribution
    :param beta: parameter of the prior distribution
    :param S: calculated parameter for the posterior distribution
    :param F: calculated parameter for the posterior distribution
    :param delta: the parameter for the optimal arm distribution
    :param num_steps: number of times to run the algorithm
    :return: returns the statistics after the run
    """
    S = np.zeros(K)
    F = np.zeros(K)
    theta = np.zeros(K)
    for _ in range(num_steps):
        for i in range(K):
            theta[i] = np.random.beta(alpha + S[i], beta + F[i])
        arm_idx = np.argmax(theta)
        reward = arm_utils.get_reward(arm_idx == statistics.optimal_arm, delta)
        if reward == 1:
            S[arm_idx] += 1
        else:
            F[arm_idx] += 1
        statistics.update_statistics(arm_idx, reward)

    return statistics


import numpy as np
from scipy.stats import bernoulli


def get_reward(is_optimal, delta):
    """
    This function will return a stochastic reward
    :param is_optimal: is the arm the optimal arm, if so p=0.5+delta else p=0.5
    :param delta: parameter for the optimal arm reward probability
    :return: the reward
    """
    if is_optimal:
        p = 0.5 + delta
    else:
        p = 0.5
    return bernoulli.rvs(p)


def get_ucbs(averages, nums, T):
    """
    convert the history of rewards to a list of ucbs of the arms
    :param averages: averages of rewards for each of the arms
    :return: list of ucbs
    """
    K = len(averages)
    ucbs = np.zeros(K)
    ucbs[nums!=0] = (np.sqrt(2 * np.log(T) / nums[nums!=0]) + averages[nums!=0])
    ucbs[nums == 0] = np.inf
    """
    for i in range(K):
        if nums[i]==0:
            ucbs[i] = np.inf
        else:
            ucbs[i] = (np.sqrt(2 * np.log(T) / nums[i]) + averages[i])
    """
    return ucbs


def get_Bs(t, averages, em_vars, nums, c=1, b=1):
    """
    convert the history of rewards to a list of Bs of the arms
    :param t: current time step
    :param averages: averages of rewards for each of the arms
    :param em_vars: empirical variance of each arm
    :param nums: number of times each arm was chosen
    :param c: parameter of the algorithm
    :param b: parameter of the algorithm
    :return: list of Bs
    """

    K = len(averages)
    Bs = np.zeros(K)
    Bs[nums != 0] = averages[nums != 0] + np.sqrt(2*em_vars[nums != 0]*np.log(t)/nums[nums != 0]) + c*3*b*np.log(t)/nums[nums != 0]
    Bs[nums == 0] = np.inf

    return Bs

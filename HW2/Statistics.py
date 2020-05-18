import numpy as np
import copy

class Statistics:
    """
    This class will help handle all of the means and variances of the arms rewards
    """
    def __init__(self, K, optimal_arm):

        self.t = 0
        self.history = []
        self.sums = np.zeros(K)
        self.sums_squared = np.zeros(K)
        self.nums = np.zeros(K)
        self.averages = np.full(K, np.inf)
        self.em_vars = np.full(K, np.inf)
        self.optimal_arm = optimal_arm
        self.explore_index = []

    def update_statistics(self, arm_idx, reward):
        self.history.append((np.uint8(arm_idx), bool(reward)))
        self.t += 1

        if self.t == 1:
            self.explore_index.append(0)
        else:
            if self.averages[arm_idx] < np.max(self.averages):
                self.explore_index.append(self.explore_index[-1] + 1)
            else:
                self.explore_index.append(self.explore_index[-1])

        self.sums[arm_idx] += reward
        self.sums_squared[arm_idx] += reward ** 2
        self.nums[arm_idx] += 1
        self.averages[arm_idx] = self.sums[arm_idx] / self.nums[arm_idx]
        self.em_vars[arm_idx] = self.sums_squared[arm_idx]/self.nums[arm_idx] - self.averages[arm_idx] ** 2



    """
    def get_explore_index(self):
        T = len(self.history)
        explore_index = np.zeros(T)
        empirical_averages,_ = self.get_all_mean_var()
        for t, step in enumerate(self.history):
            curr_optimal = np.argmax(empirical_averages[t])
            if t == 0:
                if step[0] != curr_optimal:  # non-greedy
                    explore_index[0] = 1
                continue
            if step[0] != curr_optimal:
                explore_index[t] = explore_index[t-1] + 1
            else:
                explore_index[t] = explore_index[t-1]
        return explore_index
    """

    def get_regret(self, delta):
        T = len(self.history)
        explore_index = np.zeros(T)
        for t, step in enumerate(self.history):
            if t == 0:
                if step[0] != self.optimal_arm:  # non-greedy
                    explore_index[0] = delta
                continue
            if step[0] != self.optimal_arm:
                explore_index[t] = explore_index[t - 1] + delta
            else:
                explore_index[t] = explore_index[t - 1]
        return np.asarray(explore_index)
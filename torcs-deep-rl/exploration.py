# Stochastic processes for state exploration
import random
import numpy as np


class OrnsteinUhlenbeck():
    """
    Mean-reverting Ornstein-Uhlenbeck process,
    employed by Lillicrap et al.; ICLR 2016
    for deep reinforcement learning:
    https://arxiv.org/pdf/1509.02971.pdf
    """
    def run(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

import numpy as np


class Discrete(object):
    """
    {0,1,...,n-1}
    inspired by the Discrete class of the gym package.

    Example:
        env.observation_space = Discrete(2)
        env.action_space.sample()
    """
    def __init__(self, n):
        self.n = n
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        return self.np_random.randint(self.n)

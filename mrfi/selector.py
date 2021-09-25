import operator
from functools import reduce
import scipy.stats
import numpy as np

class EmptySelector:
    def gen_list(self, shapes):
        return []

class RandomPositionSelector_FixN:
    def __init__(self, n: int):
        self.n = n
    
    def gen_list(self, shape):
        if len(shape) > 1:
            shape=reduce(operator.mul, shape)
        return np.random.randint(0, shape, self.n)

class RandomPositionSelector_Rate:
    def __init__(self, rate: float, poisson: bool = True):
        self.rate=rate
        self.poisson = poisson

    def gen_list(self, shape):
        if len(shape) > 1:
            shape=reduce(operator.mul, shape)

        n=int(shape * self.rate)
        if self.poisson:
            n=scipy.stats.poisson.rvs(n)
        else:
            n=int(n)
        return np.random.randint(0, shape, n)

Selector_Dict = {
    None: EmptySelector,
    'RandomPositionSelector_FixN': RandomPositionSelector_FixN,
    'RandomPositionSelector_Rate': RandomPositionSelector_Rate,
}

import operator
from functools import reduce
from typing import List
import scipy.stats
import numpy as np

class EmptySelector:
    def gen_list(self, shapes):
        return []

class FixPositionSelector:
    def __init__(self, pos: List[int]):
        self.pos = sorted(pos)
    
    def gen_list(self, shape):
        if len(shape) > 1:
            shape=reduce(operator.mul, shape)
        assert(len(self.pos)==0 or self.pos[-1]<shape)
        return self.pos


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
        #print(shape, end=' ')
        if len(shape) > 1:
            shape=reduce(operator.mul, shape)

        n=shape * self.rate # ! rate should not be str
        #print(n)
        if self.poisson:
            n=scipy.stats.poisson.rvs(n)
        else:
            n=int(n)
        return np.random.randint(0, shape, n)

Selector_Dict = {
    None: EmptySelector,
    'RandomPositionSelector_FixN': RandomPositionSelector_FixN,
    'RandomPositionSelector_Rate': RandomPositionSelector_Rate,
    'FixPositionSelector': FixPositionSelector,
}

import operator
from functools import reduce
from typing import List
import scipy.stats
import torch

def EmptySelector(shapes):
    return []

def FixPosition(shape, position: List[int]):
    nelem = shape.numel()
    assert len(position)==0 or position[-1]<nelem
    return position

def RandomPositionByNumber(shape, n):
    nelem = shape.numel()
    return torch.randint(0, nelem, (n,))

def RandomPositionByRate(shape, rate: float, poisson_sample: bool = True):
    nelem = shape.numel()
    # print(nelem, rate)

    n=nelem * float(rate)
    #print(n)
    if poisson_sample:
        n=scipy.stats.poisson.rvs(n)
    else:
        n=int(n)
    return torch.randint(0, nelem, (n,))

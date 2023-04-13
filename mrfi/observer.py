from typing import Callable
import numpy as np
import torch.nn as nn
import torch

class MinMax:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.min = self.max = None

    def update(self, x, golden):
        minv = torch.min(x).item()
        maxv = torch.max(x).item()
        if self.min is None:
            self.min = minv
        else:
            self.min = min(self.min, minv)

        if self.max is None:
            self.max = maxv
        else:
            self.max = maxv
    
    def result(self):
        return self.min, self.max

class RMSE:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.golden_act = None
        self.last_is_golden = False
        self.MSE_sum = []

    def update(self, x, golden):
        if golden:
            self.last_is_golden = True
            self.golden_act = x
        else:
            if not self.last_is_golden:
                raise ValueError('RMSE observer require golden run before FI run')
            
            mse = torch.sum((x-self.golden_act)**2)/x.numel()
            self.MSE_sum.append(mse.item())

            self.last_is_golden = False
            self.golden_act = None
    
    def result(self):
        return np.sqrt(np.sum(self.MSE_sum)/len(self.MSE_sum))

def mapper_identity(x, golden):
    return x

def mapper_maxabs(x, golden):
    return np.max(np.abs(x))

def mapper_minmax(x, golden):
    return np.array(np.min(x), np.max(x))

def mapper_var(x, golden):
    return np.var(x)

def mapper_diff(x, golden):
    return x-golden

def mapper_maxdiff(x, golden):
    return np.max(x-golden, axis=1)

def mapper_sumdiff(x, golden):
    return np.sum(x-golden, axis=1)
    
def mapper_mse(x, golden):
    return np.mean((x-golden)**2, axis=1)

def mapper_sse(x, golden):
    return np.sum((x-golden)**2, axis=1)

def mapper_mae(x, golden):
    return np.mean(np.abs(x-golden), axis=1)

def mapper_sae(x, golden):
    return np.sum(np.abs(x-golden), axis=1)

def mapper_accurancy(x, golden):
    return np.all(x==golden, axis=1)

def mapper_equalcount(x, golden):
    return np.sum(x==golden, axis=1)

def mapper_equalrate(x, golden):
    return np.mean(x==golden, axis=1)

def no_reduce(arg):
    if isinstance(arg, list):
        return np.concatenate(arg)
    return arg

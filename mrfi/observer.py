from typing import Callable
import numpy as np
import torch.nn as nn

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
    
Mapper_Dict = {
    'identity': mapper_identity,
    'maxabs': mapper_maxabs,
    'minmax': mapper_minmax,
    'var': mapper_var,
    'diff': mapper_diff,
    'maxdiff': mapper_maxdiff,
    'sumdiff': mapper_sumdiff,
    'mse': mapper_mse,
    'sse': mapper_sse,
    'mae': mapper_mae,
    'sae': mapper_sae,
    'accurancy': mapper_accurancy,
    'equalcount': mapper_equalcount,
    'equalrate': mapper_equalrate,
}

Reducer_Dict = {
    'no_reduce': np.append,
    'sum': np.sum,
    'max': np.max,
    'min': np.min,
}

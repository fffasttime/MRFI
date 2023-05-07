import operator
from functools import reduce
from typing import List, Union
import scipy.stats
import torch
import numpy as np
import logging

def _flatten_position(shape, position: torch.Tensor):
    assert len(shape) == len(position), \
        f'position should have same Dimision as value tensor shape {shape}({len(shape)}D), got {len(position)}D position list'
    flat_pos = position[0]
    if len(shape) > 1:
        flat_pos = flat_pos * shape[1] + position[1]
    if len(shape) > 2:
        flat_pos = flat_pos * shape[2] + position[2]
    if len(shape) > 3:
        flat_pos = flat_pos * shape[3] + position[3]
    return flat_pos

def EmptySelector(shape):
    return []

def FixPosition(shape, position: Union[int, List[int]]):
    pos = torch.tensor(position, dtype=torch.long)
    if len(pos) == 1:
        nelem = shape.numel()
        assert 0 <= pos[0] < nelem
        return pos
    else:
        return _flatten_position(shape, pos)

def FixPositions(shape, positions: Union[List[int], List[List[int]]]):
    pos = torch.tensor(positions, dtype=torch.long)
    if len(pos.shape) == 1:
        nelem = shape.numel()
        assert len(positions)==0 or positions[-1]<nelem
        return pos
    else:
        assert len(pos.shape) == 2
        return _flatten_position(shape, pos)

def RandomPositionByNumber(shape, n = 1):
    nelem = shape.numel()
    return torch.randint(0, nelem, (n,))

def _get_num_by_rate(shape, rate, poisson_sample):
    nelem = shape.numel()

    n=nelem * float(rate)
    if poisson_sample:
        n=scipy.stats.poisson.rvs(n)
    else:
        n=int(round(n))
    
    logging.debug('selector: Random num at shape %s, %d elem, %d selected'%(str(shape), nelem, n))
    return n

def _check_rate_zero(rate):
    if 0 < rate < 0.1: 
        return False
    if rate == 0: 
        return True
    if 1 > rate > 0.1: 
        logging.warning('selector: Value select rate is too large (%f), may get inaccuracy result'%rate)
        return False
    raise ValueError('Invalid error rate: %f'%(rate))

def RandomPositionByRate(shape, rate: float = 1e-4, poisson_sample: bool = True):
    rate = float(rate)
    if _check_rate_zero(rate): return []
    nelem = shape.numel()
    n = _get_num_by_rate(shape, rate, poisson_sample)
    return torch.randint(0, nelem, (n,))

def _get_mask_kwargs(shape, kwargs: dict):
    '''
    For common pytorch memory layout:
    CNN feature map: (instance, channel, height, weight)
    CNN weight: (out_channel, in_channel, height, weight)
    Full connect layer:  (instance, neuron)
    Full connect weight:  (out, in)
    '''
    d0 = kwargs.get('instance') or kwargs.get('out_channel') or kwargs.get('d0') or kwargs.get('out')
    d1 = kwargs.get('in_channel') or kwargs.get('channel') or kwargs.get('d1') or kwargs.get('in') or kwargs.get('neuron')
    d2 = kwargs.get('height') or kwargs.get('d2')
    d3 = kwargs.get('width') or kwargs.get('d3')
    return d0, d1, d2, d3

def _get_pos_with_mask(shape, n, dimmasks, inverse = True):
    pos = torch.empty((len(shape), n), dtype=torch.long)
    for dim, dimsize in enumerate(shape):
        if dimmasks[dim] is not None:
            if inverse:
                used = torch.from_numpy(np.setdiff1d(np.arange(dimsize), np.array(dimmasks[dim])))
            else:
                used = torch.from_numpy(np.array(dimmasks[dim]))
            if len(used) == 0:
                logging.warning('selector: No FI position selected after mask/select, please check')
                return []
            idx = torch.randint(0, len(used), (n,))
            pos[dim] = used[idx]
        else:
            pos[dim] = torch.randint(0, dimsize, (n,))
    
    return _flatten_position(shape, pos)

def MaskedDimRandomPositionByNumber(shape, n = 1, **kwargs):
    dimmasks = _get_mask_kwargs(shape, kwargs)
    return _get_pos_with_mask(shape, n, dimmasks)

def SelectedDimRandomPositionByNumber(shape, n = 1, **kwargs):
    dimmasks = _get_mask_kwargs(shape, kwargs)
    return _get_pos_with_mask(shape, n, dimmasks, False)

def MaskedDimRandomPositionByRate(shape, rate: float, poisson_sample: bool = True, **kwargs):
    rate = float(rate)
    if _check_rate_zero(rate): return []
    dimmasks = _get_mask_kwargs(shape, kwargs)
    rate_reduce = float(rate)
    for i, dimsize in enumerate(shape):
        if dimmasks[i] is not None:
            rate_reduce = rate * (1 - len(dimmasks[i])/dimsize)
    return _get_pos_with_mask(shape, _get_num_by_rate(shape, rate_reduce, poisson_sample), dimmasks)

def SelectedDimRandomPositionByRate(shape, rate: float, poisson_sample: bool = True, **kwargs):
    rate = float(rate)
    if _check_rate_zero(rate): return []
    dimmasks = _get_mask_kwargs(shape, kwargs)
    rate_reduce = float(rate)
    for i, dimsize in enumerate(shape):
        if dimmasks[i] is not None:
            rate_reduce = rate * (len(dimmasks[i])/dimsize)
    return _get_pos_with_mask(shape, _get_num_by_rate(shape, rate_reduce, poisson_sample), dimmasks, False)

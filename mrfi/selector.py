"""MRFI selector methods

A selector is a call back function with fixed argument `shape`, 
and other optional args specified in config file.

The selector determines which positions of tensor are send to fault injector i.e. `error_mode`.

Selector returns a 1-d index tensor with type `torch.long`, 
indicates the index of target tensor after flatten.

Note:
    If no selector specified in config file, all target tensor will be sent to fault injector.
"""

import logging
from typing import List, Union

import numpy as np
import scipy.stats
import torch

def _flatten_position(shape, position: torch.Tensor):
    assert len(shape) == len(position), \
        f"Position should have same Dimision as value tensor shape {shape}({len(shape)}D),"\
         " got {len(position)}-D position list"
    flat_pos = position[0]
    if len(shape) > 1:
        flat_pos = flat_pos * shape[1] + position[1]
    if len(shape) > 2:
        flat_pos = flat_pos * shape[2] + position[2]
    if len(shape) > 3:
        flat_pos = flat_pos * shape[3] + position[3]
    return flat_pos

def EmptySelector(shape):
    """No position selected by this, for debug or special use."""
    return []

def FixPosition(shape, position: Union[int, List[int]]):
    """Select fixed *one* position by coordinate.
    
    Args:
        position: 
            if `int`, stands for index of . 
            if `List[int]`, stands for n-d coordinate of target position.
    """
    pos = torch.tensor(position, dtype=torch.long)
    if len(pos) == 1:
        nelem = shape.numel()
        assert 0 <= pos[0] < nelem
        return pos
    else:
        return _flatten_position(shape, pos)

def FixPositions(shape, positions: Union[List[int], List[List[int]]]):
    """Select a list of fixed positions by coordinate.
    
    Args:
        positions: 
            if `List[int]`, stands for index of target tensor after flatten 
            if `List[List[int]]`, stands for n-d coordinate of target position.
    """
    pos = torch.tensor(positions, dtype=torch.long)
    if len(pos.shape) == 1:
        nelem = shape.numel()
        assert len(positions)==0 or positions[-1]<nelem
        return pos
    else:
        assert len(pos.shape) == 2
        return _flatten_position(shape, pos)

def RandomPositionByNumber(shape, n: int = 1):
    """Select n random positions.
    
    Args:
        n: Number of target positions.
    """
    nelem = shape.numel()
    return torch.randint(0, nelem, (int(n),))

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
    if 0.1 < rate < 1:
        logging.warning('selector: Value select rate is too large (%f), may get inaccuracy result', rate)
        return False
    raise ValueError('Invalid error rate: %f'%(rate))

def RandomPositionByRate(shape, rate: float = 1e-4, poisson_sample: bool = True):
    """Select random positions by rate.
    
    Args:
        rate: Rate of each position to be chosen.
        poisson_sample: Enable poisson samping, which is more accurate when rate is quite small.

    Info:
        `rate` stands for tensor value selected rate here. To calculate the bit error rate on bit flip experiment,
        it needs to be additionally divided by the bit width.

    """
    rate = float(rate)
    if _check_rate_zero(rate): return []
    nelem = shape.numel()
    n = _get_num_by_rate(shape, rate, poisson_sample)
    return torch.randint(0, nelem, (n,))

def _get_mask_kwargs(shape, kwargs: dict):
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


def MaskedDimRandomPositionByNumber(shape, n: int = 1, **kwargs):
    """Select n positions after specifed dimensions are masked.
    
    For a 2-d tensor, it is equivalent to selecting on a submatrix
    where some rows and cols are masked.

    Info:
        MaskedDim- and SelectedDim- selectors can be used for fine-grained 
        evaluate and selective protect experiments, 
        including instance-wise, channel-wise and spatial-wise.
    Tip:
        Mask argeuments follow common pytorch memory layout:\n
        - CNN feature map: (`instance`, `channel`, `height`, `width`)\n
        - CNN weight: (`out_channel`, `in_channel`, `height`, `width`)\n
        - Full connect layer:  (`instance`, `neuron`)\n
        - Full connect weight:  (`out`, `in`)
    Args:
        n (int): number of position to select.
        **kwargs:
            instance `list[int]`: dim = 0\n
            channel `list[int]`: dim = 1\n
            height `list[int]`: dim = 2\n
            width `list[int]`: dim = 3\n
            out_channel `list[int]`: dim = 0\n
            in_channel `list[int]`: dim = 1\n
            out `list[int]`: dim = 0\n
            in `list[int]`: dim = 1\n
            neuron `list[int]`: dim = 1
    """
    dimmasks = _get_mask_kwargs(shape, kwargs)
    return _get_pos_with_mask(shape, n, dimmasks)

def SelectedDimRandomPositionByNumber(shape, n: int = 1, **kwargs):
    """Select n positions on selected coordinate.

    For argument list, please refer `MaskedDimRandomPositionByNumber`.\n
    Note if one dimension selection list is not specified, 
    it stands for all of this dimension are possible selected.
    """
    dimmasks = _get_mask_kwargs(shape, kwargs)
    return _get_pos_with_mask(shape, n, dimmasks, False)

def MaskedDimRandomPositionByRate(shape, rate: float, poisson_sample: bool = True, **kwargs):
    """Select by rate where some coordinate are masked.

    For argument list, please refer `MaskedDimRandomPositionByNumber`.
    """
    rate = float(rate)
    if _check_rate_zero(rate): return []
    dimmasks = _get_mask_kwargs(shape, kwargs)
    rate_reduce = float(rate)
    for i, dimsize in enumerate(shape):
        if dimmasks[i] is not None:
            rate_reduce = rate * (1 - len(dimmasks[i])/dimsize)
    return _get_pos_with_mask(shape, _get_num_by_rate(shape, rate_reduce, poisson_sample), dimmasks)

def SelectedDimRandomPositionByRate(shape, rate: float, poisson_sample: bool = True, **kwargs):
    """Select on some coordinate by rate.

    For argument list, please refer `MaskedDimRandomPositionByNumber`.\n
    Note if one dimension selection list is not specified, 
    it stands for all of this dimension are possible selected.
    """
    rate = float(rate)
    if _check_rate_zero(rate): return []
    dimmasks = _get_mask_kwargs(shape, kwargs)
    rate_reduce = float(rate)
    for i, dimsize in enumerate(shape):
        if dimmasks[i] is not None:
            rate_reduce = rate * (len(dimmasks[i])/dimsize)
    return _get_pos_with_mask(shape, _get_num_by_rate(shape, rate_reduce, poisson_sample), dimmasks, False)

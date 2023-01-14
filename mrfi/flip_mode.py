from typing import Callable
import numpy as np
import torch

def flip_int_highest(x_in, bit_width):
    x=x_in.int()
    return (x ^ torch.full_like(x, -(1<<bit_width-1))).to(x_in.dtype)

def flip_int_random(x_in, bit_width):
    x=x_in.int()
    bit = torch.randint_like(x, 0, bit_width)
    bitmask = torch.ones_like(x) << bit
    bitmask[bit == bit_width-1] = - bitmask[bit == bit_width-1]
    return (x ^ bitmask).to(x_in.dtype)

def flip_int_fixbit(x_in, bit_width, bit):
    x=x.int()
    bitmask = 1 << bit
    if bit == bit_width-1:
        bitmask = - bitmask
    return (x ^ torch.full_like(x, bitmask)).to(x_in.dtype)

def set_0(x):
    return torch.zeros_like(x)

def set_value(x, value):
    return torch.full_like(x, value)

FlipMode_Dict={
    None: None,
    'flip_int_highest': flip_int_highest,
    'flip_int_random': flip_int_random,
    'flip_int_fixbit': flip_int_fixbit,
    'set_0': set_0,
    'set_value': set_value,
}

def add_custom_flip_mode(name: str, func: Callable):
    FlipMode_Dict[str] = func

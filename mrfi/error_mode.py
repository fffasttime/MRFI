import numpy as np
import torch

def IntSignBitFlip(x_in, bit_width):
    x=x_in.int()
    return (x ^ torch.full_like(x, -(1<<bit_width-1))).to(x_in.dtype)

def IntRandomBitFlip(x_in, bit_width):
    x=x_in.int()
    bit = torch.randint_like(x, 0, bit_width)
    bitmask = torch.ones_like(x) << bit
    bitmask[bit == bit_width-1] = - bitmask[bit == bit_width-1]
    return (x ^ bitmask).to(x_in.dtype)

def IntFixedBitFlip(x_in, bit_width, bit):
    x=x.int()
    bitmask = 1 << bit
    if bit == bit_width-1:
        bitmask = - bitmask
    return (x ^ torch.full_like(x, bitmask)).to(x_in.dtype)

def SetZero(x):
    return torch.zeros_like(x)

def SetValue(x, value):
    return torch.full_like(x, value)

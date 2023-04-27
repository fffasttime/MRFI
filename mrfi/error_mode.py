import numpy as np
import torch
from typing import Union, Sized, Optional

def IntSignBitFlip(x_in: torch.Tensor, bit_width: int):
    x = x_in.int()
    return (x ^ torch.full_like(x, -(1<<bit_width-1))).to(x_in.dtype)

def IntRandomBitFlip(x_in: torch.Tensor, bit_width: int):
    x = x_in.int()
    bit = torch.randint_like(x, 0, bit_width)
    bitmask = torch.ones_like(x) << bit
    bitmask[bit == bit_width-1] = - bitmask[bit == bit_width-1]
    return (x ^ bitmask).to(x_in.dtype)

def IntFixedBitFlip(x_in: torch.Tensor, bit_width: int, bit: Union[int, Sized]):
    x = x_in.int()
    if hasattr(bit, '__len__'):
        bit = torch.tensor(bit, dtype=torch.int, device=x.device)
        assert x.shape == bit.shape, f'when `bit` arg is list, its shape should same as selected value, expect {x.shape}, got {bit.shape}'
        bitmask = 1 << bit
        signbit = torch.where(bit == bit_width - 1)
        bitmask[signbit] = -bitmask[signbit]
        return (x ^ bitmask).to(x_in.dtype)
    else:
        bit = int(bit)
        assert 0 <= bit < bit_width
        bitmask = 1 << bit
        if bit == bit_width-1:
            bitmask = - bitmask
        return (x ^ torch.full_like(x, bitmask)).to(x_in.dtype)

def SetZero(x: torch.Tensor):
    return torch.zeros_like(x)

def SetValue(x: torch.Tensor, value: Union[int, float, Sized]):
    if hasattr(value, '__len__'):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)
        assert x.shape == value.shape, f'when `value` arg is a list, its shape should same as selected value, expect {x.shape}, got {value.shape}'
        return value
    return torch.full_like(x, value)

def _get_float_type(x_in, floattype = None):
    if floattype is None:
        floattype = x_in.dtype
    if floattype == torch.float32 or floattype == np.float32:
        bit_width = 32
        nptype = np.float32
        npinttype = np.int32
    elif floattype == torch.float64 or floattype == np.float64:
        bit_width = 64
        nptype = np.float64
        npinttype = np.int64
    elif floattype == torch.float16 or floattype == np.float16:
        bit_width = 16
        nptype = np.float16
        npinttype = np.int16
    else:
        raise TypeError("Unknown float type '%s'"%floattype)
    
    return bit_width, nptype, npinttype

def FloatRandomBitFlip(x_in: torch.Tensor, 
                       floattype: Optional[Union[np.float16, np.float32, np.float64]] = None):
    bit_width, nptype, npinttype = _get_float_type(x_in, floattype)
    bit = torch.randint_like(x_in, 0, bit_width)
    bitmask = 1 << bit.cpu().numpy().astype(npinttype)
    
    np_value = x_in.cpu().numpy()
    if nptype != np_value.dtype: 
        np_value = np_value.astype(nptype)
    np_value.dtype = npinttype # trickly change type

    np_value ^= bitmask
    np_value.dtype = nptype

    return torch.tensor(np_value, dtype=x_in.dtype, device=x_in.device)

def FloatFixBitFlip(x_in: torch.Tensor, 
                    bit: Union[int, Sized], 
                    floattype: Optional[Union[np.float16, np.float32, np.float64]] = None):
    bit_width, nptype, npinttype = _get_float_type(x_in, floattype)

    if hasattr(bit, '__len__'):
        bit = np.array(bit, dtype=npinttype)
        assert len(x_in) == len(bit), f'when `bit` arg is a list, its shape should same as selected value, expect {x_in.shape}, got {bit.shape}'
    else:
        assert 0 <= bit < bit_width

    ones = torch.ones_like(x_in, device='cpu').numpy().astype(npinttype)
    bitmask = ones << bit
    
    np_value = x_in.cpu().numpy()
    if nptype != np_value.dtype: 
        np_value = np_value.astype(nptype)
    np_value.dtype = npinttype # trickly change type

    np_value ^= bitmask
    np_value.dtype = nptype

    return torch.tensor(np_value, dtype=x_in.dtype, device=x_in.device)

def IntRandom(x_in, low: int = 0, high: int = 128):
    return torch.randint_like(x_in, low, high)

def IntRandomBit(x_in, bit_width: int = 8, signed: int = False):
    if signed:
        assert bit_width > 1
        return torch.randint_like(x_in, -2**(bit_width-1), 2**(bit_width-1))
        
    return torch.randint_like(x_in, 0, 2**bit_width)

def UniformRandom(x_in, low = -1, high = 1):
    return torch.rand_like(x_in) * (high - low) + low

def NormalRandom(x_in, mean = 0, std = 1):
    return torch.randn_like(x_in) * std + mean

def UniformDisturb(x_in, low = -1, high = 1):
    return torch.rand_like(x_in) * (high - low) + low + x_in

def NormalDisturb(x_in, std = 1, bias = 0):
    return torch.randn_like(x_in) * std + bias + x_in

"""MRFI error mode methods, i.e. fault injectors

A error_mode callback function recieve a fixed argument `x_in`,
and other arguments specified by config, 
then *return* a tensor with values after fault inject. 
DO NOT modify data in tensor `x_in`.

`x_in` is a 1-d tensor contains values selected by `selector`.
The return fault injected tensor should have same shape and type as x_in.

Tip:
    The fault injection are implemented by pytorch functions, 
    so they are both suitable on CPUs and GPUs.\n
    Use `fi_model = MRFI(NNModel().cuda(), configs)` and use `cuda` tensor
    can automatically perform error injection on GPU.

Note:
    - Integer fault injectors usually REQUIRES quantization.
    - Float bit flip injectors usually NOT use quantization. 
"""
from typing import Optional, Sized, Union

import numpy as np
import torch

def IntSignBitFlip(x_in: torch.Tensor, bit_width: int):
    """Flip integer tensor on sign bit
    
    Args:
        bit_width: bit width of quantized value.
    """
    x = x_in.int()
    return (x ^ torch.full_like(x, -(1<<bit_width-1))).to(x_in.dtype)

def IntRandomBitFlip(x_in: torch.Tensor, bit_width: int):
    """Flip integer tensor on random bit
    
    Args:
        bit_width: bit width of quantized value.
    """
    x = x_in.int()
    bit = torch.randint_like(x, 0, bit_width)
    bitmask = torch.ones_like(x) << bit
    bitmask[bit == bit_width-1] = - bitmask[bit == bit_width-1]
    return (x ^ bitmask).to(x_in.dtype)

def IntFixedBitFlip(x_in: torch.Tensor, bit_width: int, bit: Union[int, Sized]):
    """Flip integer tensor on specified bit
    
    if `bit` is a list, it should have same length as `x_in`.
    Can cooperate with a fixed number selector.
    
    Args:
        bit_width: bit width of quantized value.
        bit:
            If `bit` is an integer 0<=`bit`<`bit_width`, flip on fixed bit.

            If `bit` is a list of integer, the list should have the same length 
            as selected tensor values. Then this function flip each corresponding bit.
    """
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
    """Fault inject selected value to zero"""
    return torch.zeros_like(x)

def SetValue(x: torch.Tensor, value: Union[int, float, Sized]):
    """Fault inject value to specified value(s)
    
    if `value` is a list, it should have same length as input.
    Can cooperate with a fixed number selector.
    
    Args:
        value:
            If `value` is an scalar value(i.e. int, float), 
            set selected tensor to value.

            If `value` is a list, 
            the list should have the same length as selected tensor values. 
            Then this function set selected value according to the list.
    """
    if hasattr(value, '__len__'):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)
        assert x.shape == value.shape, f'when `value` arg is a list, its shape should same as selected value, expect {x.shape}, got {value.shape}'
        return value
    return torch.full_like(x, value)

def _get_float_type(x_in, floattype = None):
    if floattype is None:
        floattype = x_in.dtype
    if floattype == torch.float32 or floattype == np.float32 or floattype == 'float32':
        bit_width = 32
        nptype = np.float32
        npinttype = np.int32
    elif floattype == torch.float64 or floattype == np.float64 or floattype == 'float64':
        bit_width = 64
        nptype = np.float64
        npinttype = np.int64
    elif floattype == torch.float16 or floattype == np.float16 or floattype == 'float16':
        bit_width = 16
        nptype = np.float16
        npinttype = np.int16
    else:
        raise TypeError("Unknown float type '%s'"%str(floattype))
    
    return bit_width, nptype, npinttype

def FloatRandomBitFlip(x_in: torch.Tensor, 
                       floattype: Optional[Union[np.float16, np.float32, np.float64]] = None,
                       suppress_invalid: bool = False):
    """Flip bit randomly on float values
    
    Flip random bit when regards the values as IEEE-754 float type.
    
    Args:
        floattype: 
            Should be one of "float16", "float32", "float64".
            if `None`, use the input tensor type, usually is "float32".
        
        suppress_invalid:
            if `True`, set flipped inf/nan value to 0.
    """
    bit_width, nptype, npinttype = _get_float_type(x_in, floattype)
    bit = torch.randint_like(x_in, 0, bit_width)
    bitmask = 1 << bit.cpu().numpy().astype(npinttype)
    
    np_value = x_in.cpu().numpy()
    if nptype != np_value.dtype: 
        np_value = np_value.astype(nptype)
    np_value.dtype = npinttype # trickly change type

    np_value ^= bitmask
    np_value.dtype = nptype

    res = torch.tensor(np_value, dtype=x_in.dtype, device=x_in.device)

    if suppress_invalid:
        res[torch.isnan(res)] = 0
        res[torch.isinf(res)] = 0

    return res

def FloatFixedBitFlip(x_in: torch.Tensor, 
                    bit: Union[int, Sized], 
                    floattype: Optional[Union[np.float16, np.float32, np.float64]] = None,
                    suppress_invalid: bool = False):
    """Flip specific bit on float values
    
    Flip specified bit when regards the values as IEEE-754 float type.
    
    Args:
        bit:
            If `bit` is an integer 0 <= `bit` < float bits, flip on fixed bit.

            If `bit` is a list of integer, the list should have the same length 
            as selected tensor values. Then this function flip each corresponding bit.

        floattype: 
            Should be one of "float16", "float32", "float64".
            if `None`, use the input tensor type, usually is "float32".

        suppress_invalid:
            if `True`, set flipped inf/nan value to 0.
    """
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

    res = torch.tensor(np_value, dtype=x_in.dtype, device=x_in.device)

    if suppress_invalid:
        res[torch.isnan(res)] = 0
        res[torch.isinf(res)] = 0

    return res

def IntRandom(x_in, low: int = 0, high: int = 128):
    """Uniformly set to random integer in range.
    
    Set fault injection value to integer in range `[low, high)` with uniform probability.
    
    Args:
        low: Minimum range, included.
        high: Maximum range, excluded.
    """
    low, high = int(low), int(high)
    return torch.randint_like(x_in, low, high)

def IntRandomBit(x_in, bit_width: int = 8, signed: bool = False):
    """Set fault value to a random `bit_width`-bit integer.
    
    If unsigned, randomly set to `[0, 2**bit_width)`.
    If signed, randomly set to `[-2**(bit_width-1), 2**(bit_width-1))`.
    
    Args:
        bit_width: Width of target value.
        signed: Signed range.
    """
    bit_width, signed = int(bit_width), bool(signed)
    if signed:
        assert bit_width > 1
        return torch.randint_like(x_in, -2**(bit_width-1), 2**(bit_width-1))

    return torch.randint_like(x_in, 0, 2**bit_width)

def UniformRandom(x_in, low: float = -1, high: float = 1):
    """Uniform set to float number in range.

    Set fault injection value to float number in `[low, high]` with uniform probability.
    
    Args:
        low: Minimum range.
        high: Maximum range.
    """
    low, high = float(low), float(high)
    return torch.rand_like(x_in) * (high - low) + low

def NormalRandom(x_in, mean: float = 0, std: float = 1):
    """Set fault injection value to a normal distribution.
     
    Set fault injection value to a normal distribution with specified `mean`, `std`.
    i.e. $ X ~ Normal(mean, std^2) $.

    Args:
        mean: Distribution mean.
        std: Distribution standard deviation.
    """
    mean, std = float(mean), float(std)
    return torch.randn_like(x_in) * std + mean

def UniformDisturb(x_in, low: float = -1, high: float = 1):
    """Add uniform distribution noise to current value.
    
    Add uniform distribution noise in range `[low, high]` to current value.
    
    Args:
        low: Minimum noise range.
        high: Maximum noise range.
    """
    low, high = float(low), float(high)
    return torch.rand_like(x_in) * (high - low) + low + x_in

def NormalDisturb(x_in, std: float = 1, bias: float = 0):
    """Add normal distribution noise to current value.
     
    Add a normal distribution with specified `std`, `bias`.
    i.e. $ x_{out}=x_{in} + X + bias, X ~ Normal(0, std^2) $.

    Args:
        std: Distribution mean.
        bias: Extra noise bias.
    """
    std, bias = float(std), float(bias)
    return torch.randn_like(x_in) * std + bias + x_in

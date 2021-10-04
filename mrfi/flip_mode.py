from typing import Callable
import numpy as np

def flip_int_highest(x, bit_width):
    return int(x) ^ -(1<<bit_width-1)

def flip_int_random(x, bit_width):
    bit = np.random.randint(0, bit_width)
    bitmask = 1 << bit
    if bit == bit_width-1:
        bitmask = - bitmask
    return int(x) ^ bitmask

def flip_int_fixbit(x, bit_width, bit):
    bitmask = 1 << bit
    if bit == bit_width-1:
        bitmask = - bitmask
    return int(x) ^ bitmask

def set_0(x):
    return 0

def set_value(x, value):
    return value

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

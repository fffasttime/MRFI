'''This is a test function module used for test_addfunc.py.
Will be loaded to MRFI.
'''

import torch

def TestErrorModel(x):
    return torch.full_like(x, 3)

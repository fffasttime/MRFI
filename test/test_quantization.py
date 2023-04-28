import sys

sys.path.append('.')
from mrfi import quantization
import torch

def test_SymmericQuantization():
    x = torch.arange(-1, 1, 0.1)
    
    quantization.SymmericQuantization.quantize(x, 5, 0.5) # 5-bit, [-15, 15]
    
    assert (x == torch.Tensor([-15., -15., -15., -15., -15., -15., -12.,  -9.,  -6.,  -3.,   0.,   3.,
          6.,   9.,  12.,  15.,  15.,  15.,  15.,  15.])).all()

    quantization.SymmericQuantization.dequantize(x, 5, 0.5)

    assert ((x - torch.Tensor([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.4, -0.3,
        -0.2, -0.1,  0.0,  0.1,  0.2,  0.3,  0.4,  0.5, 0.5,  0.5,  0.5,  0.5])).abs()<1e-6).all()

def test_PositiveQuantization():
    x = torch.arange(-0.5, 1, 0.1)
    
    quantization.PositiveQuantization.quantize(x, 5, 0.5) # 5-bit, [0, 31]
    
    assert (x == torch.Tensor([ 0.,  0.,  0.,  0.,  0.,  0.,  6., 12., 19., 25., 31., 31., 31., 31.,
        31.])).all()

    quantization.PositiveQuantization.dequantize(x, 5, 0.5)

    assert (x == torch.Tensor([0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
        0.0000000000, 0.0967741907, 0.1935483813, 0.3064516187, 0.4032257795,
        0.5000000000, 0.5000000000, 0.5000000000, 0.5000000000, 0.5000000000])).all()


def test_FixPointQuantization():
    x = torch.arange(-0.5, 3, 0.5)
    
    quantization.FixPointQuantization.quantize(x, 3, 4) # 3+4 bit,   1.5 -> 001 1000 -> 24
    
    assert (x == torch.Tensor([-8.,  0.,  8., 16., 24., 32., 40.])).all()

    quantization.FixPointQuantization.dequantize(x, 3, 4)

    assert (x == torch.Tensor([-0.5039370060,  0.0000000000,  0.5039370060,  1.0078740120,
         1.5118110180,  2.0157480240,  2.5196850300])).all()

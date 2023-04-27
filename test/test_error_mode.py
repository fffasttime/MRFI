import pytest
import sys
sys.path.append('.')
from mrfi import error_mode
import torch
import numpy as np

class TestErrorMode:
    def setup_class(self):
        ...
    
    def test_IntSignBitFlip_1(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.IntSignBitFlip(x_in, 8)
        assert (x == torch.Tensor([-127, -126, -125])).all()
    
    def test_IntSignBitFlip_2(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.IntSignBitFlip(x_in, 16)
        assert (x == torch.Tensor([-32767, -32766, -32765])).all()
    
    def test_IntFixedBitFlip_1(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.IntFixedBitFlip(x_in, 8, 0)
        assert (x == torch.Tensor([0, 3, 2])).all()

    def test_IntFixedBitFlip_1(self):
        x_in = torch.Tensor([1,2,3,-1])
        x = error_mode.IntFixedBitFlip(x_in, 8, 7)
        assert (x == torch.Tensor([-127, -126, -125, 127])).all()
    
    def test_IntFixedBitFlip_Vector(self):
        x_in = torch.ones(8)
        x = error_mode.IntFixedBitFlip(x_in, 8, np.arange(8))
        assert (x == torch.tensor([   0.,    3.,    5.,    9.,   17.,   33.,   65., -127.])).all()

    def test_IntRandomBitFlip(self):
        x_in = torch.randint(-128, 128, (1000,))
        x = error_mode.IntRandomBitFlip(x_in, 8)
        assert (-128<=x).all() and (x<=127).all()

    def test_SetZero(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.SetZero(x_in)
        assert (x == torch.Tensor([0, 0, 0])).all()
    
    def test_SetValue(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.SetValue(x_in, 10)
        assert (x == torch.Tensor([10, 10, 10])).all()
    
    def test_SetValue_Vector(self):
        x_in = torch.ones(8)
        x = error_mode.SetValue(x_in, list(np.arange(8)))
        assert (x == torch.arange(8)).all()
    
    def test_FloatFixBitFlip_1(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.FloatFixBitFlip(x_in, 31)
        assert (x == torch.Tensor([-1, -2, -3])).all()

    def test_FloatFixBitFlip_2(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.FloatFixBitFlip(x_in, 63, torch.float64)
        assert (x == torch.Tensor([-1, -2, -3])).all()
    
    def test_FloatFixBitFlip_3(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.FloatFixBitFlip(x_in, 15, np.float16)
        assert (x == torch.Tensor([-1, -2, -3])).all()

    def test_FloatFixBitFlip_Vector(self):
        x_in = torch.zeros(16)
        x = error_mode.FloatFixBitFlip(x_in, range(16), np.float16)
        assert (x == torch.tensor([5.9604644775e-08, 1.1920928955e-07, 2.3841857910e-07, 4.7683715820e-07,
        9.5367431641e-07, 1.9073486328e-06, 3.8146972656e-06, 7.6293945312e-06,
        1.5258789062e-05, 3.0517578125e-05, 6.1035156250e-05, 1.2207031250e-04,
        4.8828125000e-04, 7.8125000000e-03, 2.0000000000e+00, -0.0000000000e+00])).all()
    
    def test_FloatRandomBitFlip(self):
        x_in = torch.Tensor([1,2,3])
        x = error_mode.FloatRandomBitFlip(x_in, np.float64)

    def test_IntRandom(self):
        x_in = torch.ones(1000)
        x = error_mode.IntRandom(x_in, 0, 10)
        assert (0<=x).all() and (x<10).all()
    
    def test_IntRandomBit(self):
        x_in = torch.ones(1000)
        x = error_mode.IntRandomBit(x_in, 4, False)
        assert (0<=x).all() and (x<16).all()

        x_in = torch.ones(1000)
        x = error_mode.IntRandomBit(x_in, 4, True)
        assert (-8<=x).all() and (x<8).all()
    
    def test_Uniform(self):
        x_in = torch.ones(1000)
        x = error_mode.UniformRandom(x_in, -1, 1)
        assert (-1<=x).all() and (x<=1).all()

        x = error_mode.UniformDisturb(x_in, -1, 1)
        assert (0<=x).all() and (x<=2).all()

    def test_Normal(self):
        x_in = torch.ones(1000)
        x1 = error_mode.NormalRandom(x_in, mean = -1)
        x2 = error_mode.NormalDisturb(x_in, std = 1)
        
        assert x1.mean() + 1.5 < x2.mean() # I guess 99.999%+ PASS!

import sys

sys.path.append('.')
from mrfi.observer import *
import torch

def test_MinMax():
    obs = MinMax()
    obs.update(torch.Tensor([1,2,3,4]))
    assert obs.result() == (1,4)

    obs.update(torch.Tensor([2,5]))
    assert obs.result() == (1,5)

    obs.update_golden(torch.Tensor([0,3]))
    assert obs.result() == (0,5)

def test_RMSE():
    obs = RMSE()

    obs.update_golden(torch.Tensor([0, 1]))
    obs.update(torch.Tensor([1, 2]))

    assert obs.result() == 1

    obs.update_golden(torch.Tensor([0, 1]))
    assert obs.result() == 1

    obs.update(torch.Tensor([-2,-1]))
    assert obs.result() == np.sqrt(2.5)

def test_SaveLast():
    obs = SaveLast()

    obs.update_golden(torch.Tensor([0, 1]))
    obs.update(torch.Tensor([2, 3]))
    assert (obs.result()[0]==torch.Tensor([0, 1])).all() and \
        (obs.result()[1]==torch.Tensor([2, 3])).all()
    
    obs.update(torch.Tensor([4, 5]))
    assert (obs.result()[0]==torch.Tensor([0, 1])).all() and \
        (obs.result()[1]==torch.Tensor([4, 5])).all()

def test_MaxAbs():
    obs = MaxAbs()
    obs.update_golden(torch.Tensor([-7,6]))
    assert obs.result() == 7
    obs.update(torch.Tensor([1,8]))
    assert obs.result() == 8

def test_MeanABS():
    obs = MeanAbs()
    obs.update_golden(torch.Tensor([-7,6]))
    assert obs.result() == 6.5
    obs.update(torch.Tensor([1,8]))
    assert obs.result() == 5.5

def test_Std():
    obs = Std()
    obs.update_golden(torch.Tensor([2,-1]))
    assert obs.result() == np.sqrt(2.5)
    obs.update(torch.Tensor([3,4]))
    assert obs.result() == np.sqrt(7.5) # sqrt((4 + 1 + 9 + 16)/4)

def test_Shape():
    obs = Shape()
    obs.update(torch.zeros((2,3,4)))

    assert obs.result() == torch.Size((2,3,4))

def test_MAE():
    obs = MAE()
    
    obs.update_golden(torch.Tensor([0, 1]))
    obs.update(torch.Tensor([2, 4]))

    assert obs.result() == 2.5
    
    obs.update_golden(torch.Tensor([1, 1]))
    obs.update(torch.Tensor([3, 3]))

    assert obs.result() == 2.25

def test_EqualRate():
    obs = EqualRate()

    obs.update_golden(torch.Tensor([0,1,2,3,4,5]))
    obs.update(torch.Tensor([1,1,0,3,4,5]))
    assert np.allclose(obs.result(), 4/6)

    obs.update_golden(torch.Tensor([2,1,2,3,4,5]))
    obs.update(torch.Tensor([2,1,2,3,4,5]))
    assert np.allclose(obs.result(), 10/12)

def test_UniformSampling():
    UniformSampling.MAX_NUM = 100
    obs = UniformSampling()
    
    obs.update(torch.arange(10))
    assert np.all(obs.result() == np.arange(10))

    obs.update(torch.zeros(5, 5))
    assert len(obs.result()) == 35 and obs.count == 35

    obs.update_golden(torch.zeros(100))
    assert len(obs.result()) == 100 and obs.count == 135

    obs.update(torch.zeros(10000))
    assert len(obs.result()) == 100 and obs.result()[0] == 0 and obs.count == 10135

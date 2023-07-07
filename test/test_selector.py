import sys

sys.path.append('.')
from mrfi import selector
import torch

def test_EmptySelector():
    shape = torch.Size((2,2,3))
    assert selector.EmptySelector(shape) == []

def test_FixPositon():
    shape = torch.Size((2,2,3))
    flat_pos = selector.FixPosition(shape, [0,1,2])
    assert selector.FixPosition(shape, [5]) == flat_pos

def test_FixPositions():
    shape = torch.Size((2,2,3))
    flat_pos = selector.FixPositions(shape, [0,1,2,3,4,5,6])

    indexs = [[0,0,0,0,0,0,1], [0,0,0,1,1,1,0], [0,1,2,0,1,2,0]]
    assert (selector.FixPositions(shape, indexs) == flat_pos).all()

def test_RandomPositionByNumber():
    shape = torch.Size((1,64,10,20))
    assert selector.RandomPositionByNumber(shape, 100).numel() == 100
    print(selector.RandomPositionByNumber(shape, 100))

    shape = torch.Size((3,64,10,20))
    pos = selector.RandomPositionByNumber(shape, 100, True)
    assert pos.numel() == 300
    assert (pos < 64*10*20).sum() == 100
    assert (pos < 2*64*10*20).sum() == 200

def test_RandomPositionByRate():
    shape = torch.Size((1,64,10,20)) # 12800

    assert selector.RandomPositionByRate(shape, 0) == []
    assert selector.RandomPositionByRate(shape, 1e-3, False).numel() == 13

def test_RandomPositionByRate_2():
    shape = torch.Size((1,64,10,20))
    selector.RandomPositionByRate(shape, 1e-3, True).numel()

def test_RandomPositionByRate_classic():
    shape = torch.Size((1,64,100,200))
    selector.RandomPositionByRate_classic(shape, 1e-5) # 128000 * 1e-5 ~ 12.8

def test_MaskedRandomPositionByNumber():
    shape = torch.Size((2,64,10,20)) # 25600
    sel = selector.MaskedDimRandomPositionByNumber(shape, 100, instance = [0])
    assert sel.numel() == 100 and (sel>=12800).all()

def test_MaskedRandomPositionByNumber_2():
    shape = torch.Size((2,64,10,20)) # 25600
    sel = selector.MaskedDimRandomPositionByNumber(shape, 100, width = range(15))
    assert sel.numel() == 100 and (sel%20>=15).all()

def test_SelectedRandomPositionByNumber():
    shape = torch.Size((2,64,10,20)) # 25600
    sel = selector.SelectedDimRandomPositionByNumber(shape, 100, instance = [0])
    assert sel.numel() == 100 and (sel<12800).all()

def test_MaskedDimRandomPositionByRate():
    shape = torch.Size((3,64,10,20)) # 38400
    sel = selector.MaskedDimRandomPositionByRate(shape, 1e-2, False, instance = [1,2])
    assert sel.numel() == 128 and (sel<12800).all()

def test_SelectedDimRandomPositionByRate():
    shape = torch.Size((3,64,10,20)) # 38400
    sel = selector.SelectedDimRandomPositionByRate(shape, 1e-2, False, instance = [1,2])
    assert sel.numel() == 256 and (sel>=12800).all()


def test_FixedPixelByNumber():
    shape = torch.Size((2,64,4,5))
    
    target = torch.zeros(shape)
    sel = selector.FixedPixelByNumber(shape, 2, (1, 2))
    target.flatten()[sel] = 2
    assert sel.numel() == 2 and target[..., 1, 2].sum() == 4

    
    shape = torch.Size((2,64,4,5))
    target = torch.zeros(shape)
    sel = selector.FixedPixelByNumber(shape, 1, (1, 2), True)
    target.flatten()[sel] = 2

    assert sel.numel() == 2 and target[..., 1, 2].sum().int().item() == 4

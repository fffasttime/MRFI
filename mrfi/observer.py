from typing import Callable
import numpy as np
import torch.nn as nn
import torch

class BaseObserver:
    def __init__(self) -> None:
        self.reset()
    def reset(self):
        pass
    def update_golden(self, x):
        self.update(x) # By default, also update by golden run
    def update(self, x):
        pass
    def result(self):
        return None

class MinMax(BaseObserver):
    def reset(self):
        self.min = self.max = None

    def update(self, x):
        minv = torch.min(x).item()
        maxv = torch.max(x).item()
        if self.min is None:
            self.min = minv
        else:
            self.min = min(self.min, minv)

        if self.max is None:
            self.max = maxv
        else:
            self.max = max(self.max, maxv)
    
    def result(self):
        return self.min, self.max

class RMSE(BaseObserver):
    def reset(self):
        self.golden_act = None
        self.last_is_golden = False
        self.MSE_sum = []

    def update_golden(self, x):
        self.last_is_golden = True
        self.golden_act = x

    def update(self, x):
        if not self.last_is_golden:
            raise ValueError('RMSE observer require golden run before FI run')
        
        mse = torch.mean((x-self.golden_act)**2)
        self.MSE_sum.append(mse.item())

        self.last_is_golden = False
        self.golden_act = None
    
    def result(self):
        return np.sqrt(np.mean(self.MSE_sum))

class SaveLast(BaseObserver):
    def reset(self):
        self.golden_act = None
        self.fi_act = None

    def update_golden(self, x):
        self.golden_act = x.clone()

    def update(self, x):
        self.fi_act = x.clone()
    
    def result(self):
        return self.golden_act, self.fi_act
        
class MaxAbs(BaseObserver):
    def reset(self):
        self.maxabs = None

    def update(self, x):
        if self.maxabs == None:
            self.maxabs = torch.max(torch.abs(x)).item()
        else:
            self.maxabs = max(torch.max(torch.abs(x)).item(), self.maxabs)

    def result(self):
        return self.maxabs

class Std(BaseObserver):
    def reset(self):
        self.sum_var = 0
        self.n = 0

    def update(self, x):
        self.sum_var += (x**2).mean().item()
        self.n += 1

    def result(self):
        return np.sqrt(self.sum_var / self.n)

class Shape(BaseObserver):
    def reset(self):
        self.shape = None

    def update(self, x):
        self.shape = x.shape

    def result(self):
        return self.shape

class MAE(BaseObserver):
    def reset(self):
        self.MAEs = []
        self.golden_act = None
        self.last_is_golden = False

    def update_golden(self, x):
        self.last_is_golden = True
        self.golden_act = x

    def update(self, x):
        if not self.last_is_golden:
            raise ValueError('MAE observer require golden run before FI run')
        
        mae = torch.mean((x-self.golden_act).abs())
        self.MAEs.append(mae.item())

        self.last_is_golden = False
        self.golden_act = None

    def result(self):
        return np.mean(self.MAEs)

class EqualRate(BaseObserver):
    def reset(self):
        self.results = []
        self.golden_act = None
        self.last_is_golden = False

    def update_golden(self, x):
        self.last_is_golden = True
        self.golden_act = x

    def update(self, x):
        if not self.last_is_golden:
            raise ValueError('EqualRate observer require golden run before FI run')
        
        mae = torch.sum(x == self.golden_act)/x.numel()
        self.results.append(mae.item())

        self.last_is_golden = False
        self.golden_act = None

    def result(self):
        return np.mean(self.results)

class UniformSampling(BaseObserver):
    MAX_NUM = 10000
    def reset(self):
        self.data = None
        self.count = 0

    def random_sample(self, x, num):
        if num >= x.numel():
            return x.view(-1).clone()
        pos = torch.randint(0, x.numel(), (num, ))
        return x.view(-1)[pos]

    def update(self, x):
        if self.data is None:
            self.data = self.random_sample(x, self.MAX_NUM)
            self.count = x.numel()
            return
        if self.count < self.MAX_NUM: # Firstly, pad data to MAX_NUM
            padding = self.MAX_NUM - self.count
            if x.numel() < padding:
                self.data = torch.cat((self.data, x.view(-1)))
                self.count += x.numel()
                return
            self.data = torch.cat((self.data, x[:padding]))
            self.count = self.MAX_NUM
            x = x[padding:]

        self.count += x.numel()
        num = int(round(x.numel()/self.count * self.MAX_NUM))
        replace_pos = torch.randint(0, self.MAX_NUM, (num,))
        self.data[replace_pos] = self.random_sample(x, num)
    
    def result(self):
        return self.data.detach().cpu().numpy()

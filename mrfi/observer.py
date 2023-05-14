"""MRFI common used observers

Callback function `update(x)` will be called for each batch of inference.

Observer can accumulate its results between batchs, until a `reset()` is called.

Note: 
    Most of fault inject observer requires golden run before each fault inject run, 
    in order to compare the impact of fault inject.
    ```python
    # fi_model has FI observers, e.g. RMSE

    fi_model.observers_reset() 
    for inputs, labels in dataloader:
        with fi_model.golden_run():
            fi_model(inputs)
        fi_model(inputs)
    result = fi_model.observers_result()
    ```
"""
from typing import Any

import numpy as np
import torch

class BaseObserver:
    """Basement of observers.
    
    MRFI will call these callback functions when inference.
    A custom observer should implement follow functions.
    """
    def __init__(self) -> None:
        self.reset()
    def reset(self) -> None:
        """Reset observer when running multiple experiments"""
    def update_golden(self, x: torch.Tensor) -> None:
        """Callback when model inference with `mrfi.golden == True`
        
        Args:
            x: Internal observation value, usually a batched tenser of feature map.
        """
        self.update(x) # By default, also update by golden run
    def update(self, x: torch.Tensor) -> None:
        """Callback when model inference with  `mrfi.golden == False`"""
    def result(self) -> Any:
        """Callback when get observe result after experiment.
        
        You can do some postprocess of observation value here.
        """
        return None

class MinMax(BaseObserver):
    """Observe min/max range of tensors.

    Returns (tuple[float, float]):
        A tuple `(min_value, max_value)`
    """
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
    
    def result(self) -> tuple:
        return self.min, self.max

class RMSE(BaseObserver):
    """Root Mean Square Error metric between golden run and fault inject run.

    Returns (float):
        RMSE value of fault inject impact.
    """
    def reset(self):
        self.golden_act = None
        self.last_is_golden = False
        self.MSE_sum = []

    def update_golden(self, x):
        self.last_is_golden = True
        self.golden_act = x.clone()

    def update(self, x):
        if not self.last_is_golden:
            raise ValueError('RMSE observer require golden run before FI run')
        
        mse = torch.mean((x-self.golden_act)**2)
        self.MSE_sum.append(mse.item())

        self.last_is_golden = False
        self.golden_act = None
    
    def result(self) -> float:
        return np.sqrt(np.mean(self.MSE_sum))

class SaveLast(BaseObserver):
    """Simply save last inference internal tensor. This will be helpful when visualize NN feature maps.

    Returns (tuple):
        Last golden run activation and last FI run activation tuple (golden_act, FI_act).
        If no such run before get result, returns `None`.
    """
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
    """Observe max abs range of tensors.

    Returns (float):
        Similar as `x.abs().max()` but among all inference.
    """
    def reset(self):
        self.maxabs = None

    def update(self, x):
        if self.maxabs == None:
            self.maxabs = torch.max(torch.abs(x)).item()
        else:
            self.maxabs = max(torch.max(torch.abs(x)).item(), self.maxabs)

    def result(self) -> float:
        return self.maxabs

class MeanAbs(BaseObserver):
    """mean of abs, a metric of scale of values
    
    Returns (float):
        Similar as `x.abs.mean()` but among all inference.
    """
    def reset(self):
        self.sum_mean = 0
        self.n = 0

    def update(self, x):
        self.sum_mean += x.abs().mean().item()
        self.n += 1

    def result(self) -> float:
        return self.sum_mean / self.n

class Std(BaseObserver):
    """Standard deviation of zero-mean values.
    
    Returns (float):
        Similar as `sqrt((x**2).mean())` but among all inference.
    """
    def reset(self):
        self.sum_var = []
        self.n = 0

    def update(self, x):
        self.sum_var.append((x**2).mean().item())
        self.n += 1

    def result(self) -> float:
        return np.sqrt(np.mean(self.sum_var))

class Shape(BaseObserver):
    """Simply record tensor shape of last inference
    
    Returns (torch.Size):
    """
    def reset(self):
        self.shape = None

    def update(self, x):
        self.shape = x.shape

    def result(self) -> torch.Size:
        return self.shape

class MAE(BaseObserver):
    """Mean Absolute Error between golden run and fault inject run.

    Returns (float):
        MAE metric of fault inject impact.
    """
    def reset(self):
        self.MAEs = []
        self.golden_act = None
        self.last_is_golden = False

    def update_golden(self, x):
        self.last_is_golden = True
        self.golden_act = x.clone()

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
    """Compare how many value unchanged between golden run and fault inject run.

    Returns (float):
        A average ratio of how many values remain unchanged, between [0, 1].
        - If all value have changed, return 0. 
        - If all value are same as golden run, return 1.
    """
    def reset(self):
        self.results = []
        self.golden_act = None
        self.last_is_golden = False

    def update_golden(self, x):
        self.last_is_golden = True
        self.golden_act = x.clone()

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
    """Uniform sampling from tensors between all inference, up to 10000 samples.

    Co-work well with statistical visualization requirements, e.g. `plt.hist()` or `plt.boxplot()`.

    Info:
        Since feature map in NN are usually entire large, 
        save all feature map(e.g. use `SaveLast`) and sampling later is in-efficient.

        This observer automatically sampling values between all inference with uniform probability.

    Returns (np.array):
        A 1-d numpy, its length is min(all observerd values, 10000).
    """
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
    
    def result(self) -> np.array:
        return self.data.detach().cpu().numpy()

import math
import yaml
import logging
import torch
import numpy as np
from scipy.special import erf
from scipy.stats import hypergeom

import mrfi.observer
from model.vgg11 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: 64
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
sub_modules:
  features:
    sub_modules:
      0:
        FI_enable: false
      3:
      6:
      8:
      11:
      13:
      16:
      18:
observer:
    map: mse
    reduce: sum
'''

def bacc(out, label):
    return np.count_nonzero(out<out[label])/999

def bRRMSEacc(out, out_free, label):
    se = (out-out_free)**2
    # rmse(c0, clabel) + rmse(c1, clabel) + ...
    # = sqrt(se(c0)+se(clabel))/sqrt(2) + ...
    rmses = np.sqrt((se))
    rmse = np.sqrt(np.sum(rmses**2) / 1000)
    stds = np.abs(out_free - out_free[label])/2 + 0.01
    pracc = erf(1/(rmses/stds))/2 + 1/2
    return (np.sum(pracc)-pracc[label])/999

def bRRMSE(out, out_free, label):
    se = (out-out_free)**2
    # rmse(c0, clabel) + rmse(c1, clabel) + ...
    # = sqrt(se(c0)+se(clabel))/sqrt(2) + ...
    rmses = np.sqrt((se + se[label])/2)
    stds = np.abs(out_free - out_free[label])/2
    return (np.sum(rmses) - rmses[label]) /999, np.sum(stds) / 999

def exp(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    rates=np.logspace(-6, -3, 31)

    np.random.seed(0)

    for mode in [mrfi.flip_mode.flip_int_highest]:

        for ri, rate in enumerate(rates[-1:]):
            data=iter(testloader)

            for layer in FI_network.features.subinjectors:
                layer.selector_args['rate']=rate
                layer.selector_args['poisson']=True
                layer.update_selector()
                layer.flip_mode = mode

            FI_network.reset_observe_value()
            diff = []

            acc, bracc = 0, 0
            rmse, std = .0, .0
            for i in range(total):
                images, labels = next(data)
                images=images.to(device)
                out_free=FI_network(images, golden=True).cpu().numpy()[0]
                out=FI_network(images).cpu().numpy()[0]
                label = labels.numpy()[0]
                acc+=bacc(out, label)
                bracc+=bRRMSEacc(out, out_free, label)
                rr = bRRMSE(out, out_free, label)
                diff.append(out-out_free)
                rmse+=rr[0]
                std+=rr[1]
            
            observes=FI_network.get_observes()

            rrmse = rmse/std
            pracc = math.erf(1/rrmse)/2 + 0.5

            print('%.10f'%rate, end='\t')
            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), "%.4f"%(rmse/total), "%.4f"%(std/total), "%.2f%%"%(pracc*100), "%.2f%%"%(bracc/total*100), sep='\t', end='\t')
            print('%.2f%%'%(acc/total*100), flush=True)

            np.save('tempdiff.npy', diff)
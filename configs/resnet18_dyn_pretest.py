import yaml
import operator
from functools import reduce
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import mrfi.observer
import mrfi.selector
from model.resnet18 import Net, testset
from mrfi.flip_mode import flip_int_highest
from mrfi.injector import ModuleInjector
from torch import autograd

yamlcfg='''
FI_activation: false
FI_enable: true
FI_weight: true
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: auto
  dynamic_range_weight: 1
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0003
sub_modules:
  layer1:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
observer:
  map: mse
  reduce: sum
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def layerwise_quantization(x, bit_width, dynamic_range):
    down_limit, up_limit = -1<<bit_width-1, (1<<bit_width-1)-1
    x += dynamic_range
    x *= (up_limit - down_limit) / (dynamic_range*2)
    x += down_limit
    x.clamp_(down_limit, up_limit)
    x.round_()

def layerwise_dequantization(x, bit_width, dynamic_range):
    down_limit, up_limit = -1<<bit_width-1, (1<<bit_width-1)-1
    x -= down_limit
    x *= (dynamic_range*2)/(up_limit - down_limit)
    x -= dynamic_range

def experiment(total = 1000):
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True)
    net.to(device)
    net.eval()
    net_grad = Net(True)
    net_grad.to(device).eval()
    layerw = net_grad.layer1[0].conv1.weight

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    FI_network = ModuleInjector(net, config)
    data = iter(testloader)

    gradsum = np.zeros([total,2])
    
    flipw=layerw.detach().clone()
    layerwise_quantization(flipw, 8, 2)
    flipw = flip_int_highest(flipw, 8)
    layerwise_dequantization(flipw, 8, 2)
    sign = F.relu(torch.sign(flipw - layerw.detach()))

    criterion = nn.CrossEntropyLoss()
    repeat = 1000
    accg_all, acc_all = 0, 0
    for i in range(total):
        print('%3d'%i, end='\t')
        images, labels = next(data)
        images=images.to(device)
        net_grad.zero_grad()
        out = net_grad(images)
        loss = criterion(out, labels.to(device))
        grad = autograd.grad(loss, layerw, create_graph=True)[0]
        print('%7f\t%7f\t'%(torch.sum(grad.abs()).item(), torch.sum(grad.abs()*sign).item()), end='')

        FI_network.reset_observe_value()
        outg = FI_network(images, golden=True).cpu().numpy()
        acc, accg=0,0
        dloss = 0

        for j in range(repeat):
            out=FI_network(images).cpu()
            acc+=(np.argmax(out.numpy()[0])==labels.numpy()[0])
            accg+= (np.argmax(outg[0])==labels.numpy()[0])
            dloss += criterion(out, labels) - loss
    
        acc_all += acc
        accg_all += accg
        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f"%np.sqrt(value/total), end='\t')
        print("%.1f"%(acc/repeat*100), end ='\t')
        
        print("%.1f"%(accg/repeat*100), end ='\t')
        print("%f"%dloss, flush=True)
        
    print("%.2f"%(acc_all/total/repeat*100), flush=True, end ='\t')
    print("%.2f"%(accg_all/total/repeat*100), flush=True)
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
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: 8
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
  poisson: True
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
act = None

def observer_hook_func(input):
    global act
    act = input

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

    net_grad.layer1[0].conv1.register_forward_pre_hook(lambda mod, input: observer_hook_func(input[0]))
    out_values = np.zeros((total, 1000))
    
    criterion = nn.CrossEntropyLoss()
    repeat = 1
    accg_all, acc_all = 0, 0
    for i in range(total):
        print('%3d'%i, end='\t')
        images, labels = next(data)
        images=images.to(device)
        net_grad.zero_grad()
        out = net_grad(images)
        loss = criterion(out, labels.to(device))
        grad = autograd.grad(loss, act, create_graph=True)[0]
        print('%7f\t'%(torch.sum(grad.abs()).item()), end='')

        FI_network.reset_observe_value()
        outg = FI_network(images, golden=True).cpu().numpy()
        out_values[i] = outg
        acc, accg = 0, 0

        for j in range(repeat):
            out=FI_network(images).cpu().numpy()
            acc+=(np.argmax(out[0])==labels.numpy()[0])
            accg+= (np.argmax(outg[0])==labels.numpy()[0])

        acc_all += acc
        accg_all += accg
        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f"%np.sqrt(value/total), end='\t')
        print("%.1f"%(acc/repeat*100), flush=True, end ='\t')
        
        print("%.1f"%(accg/repeat*100), flush=True)

    print("%.2f"%(acc_all/total/repeat*100), flush=True, end ='\t')
    print("%.2f"%(accg_all/total/repeat*100), flush=True)
    np.save('output_logit.npy', out_values)

from random import sample
import yaml
import logging
import torch
import numpy as np
from itertools import combinations

import mrfi.observer
from model.vgg11 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: 64
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
sub_modules:
  features:
    sub_modules:
      0:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 4
      3:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 16
      6:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 32
      8:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 32
      11:
      13:
      16:
      18:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 32
observer:
    map: mse
    reduce: sum
'''


perlayer = [1.19853,0.87845,0.54152,0.4117,0.43442,0.63094,0.53303,0.36974]

def exp(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    layers=[
    getattr(FI_network.features,'0'),
    getattr(FI_network.features,'3'),
    getattr(FI_network.features,'6'),
    getattr(FI_network.features,'8'),
    getattr(FI_network.features,'11'),
    getattr(FI_network.features,'13'),
    getattr(FI_network.features,'16'),
    getattr(FI_network.features,'18'),
    ]

    np.random.seed(0)
    sel = np.zeros((56, len(layers)), dtype=int)
    combs = list(combinations(range(8),5))
    for i in range(56): 
        sel[i][list(combs[i])]=1 

    # sel = np.eye(len(layers), len(layers), dtype=int) ; sel[-1]+=1
    
    for mode in range(sel.shape[0]):
        calc = 0
        
        for i in range(len(layers)):
            layers[i].FI_enable=True
            layers[i].selector_args['rate']=0.0004*sel[mode, i]
            calc+=(perlayer[i]**2)*sel[mode, i]
            layers[i].update_selector()
        calc=np.sqrt(calc)
        
            
        print(''.join(map(str, list(sel[mode]))), end='\t')
        data=iter(testloader)
        FI_network.reset_observe_value()
        
        acc=0
        for i in range(total):
            images, labels = next(data)
            images=images.to(device)
            FI_network(images, golden=True)
            out=FI_network(images).cpu().numpy()
            acc+=(np.argmax(out[0])==labels.numpy()[0])

        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f"%np.sqrt(value/total), end='\t')
        print("%.5f"%calc, end='\t')
        print("%.2f%%"%(acc/total*100), flush=True)

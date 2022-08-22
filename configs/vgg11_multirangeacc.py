from asyncio import FastChildWatcher
from operator import mod
import yaml
import logging
import torch
import numpy as np

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

def exp(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    dynrange_base=[8,16,32,32,32,32,32,32]
    dynrange_coeff=[1,1.25,1.5,1.75,2.0]

    rates=np.logspace(-6, -3, 31)

    for mode in dynrange_coeff:
        print(mode)
        for ri, rate in enumerate(rates):
            data=iter(testloader)

            for ri, layer in enumerate(FI_network.features.subinjectors):
                layer.layerwise_quantization_dynamic_range = mode * dynrange_base[ri]
                layer.selector_args['rate']=rate
                layer.selector_args['poisson']=True
                layer.update_selector()

            FI_network.reset_observe_value()

            acc=0
            for i in range(total):
                images, labels = next(data)
                images=images.to(device)
                out_free=FI_network(images, golden=True).cpu().numpy()
                out=FI_network(images).cpu().numpy()
                acc+=(np.argmax(out[0])==labels.numpy()[0])
            
            observes=FI_network.get_observes()

            print('%.10f'%rate, end='\t')
            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), end='\t')
            print('%.2f%%'%(acc/total*100), flush=True)

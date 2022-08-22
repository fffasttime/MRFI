from random import sample
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
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: 32
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.00008
sub_modules:
  features:
    sub_modules:
      0:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 2.47
      3:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 12
      6:
      8:
      11:
        layerwise_quantization:
          bit_width: 8
          dynamic_range: 48
observer:
    map: mse
    reduce: sum
'''

perlayer = [0.26828,0.26072,0.17382,0.12126,0.1183]

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
    ]

    nexps=32
    np.random.seed(0)
    sel = np.random.randint(0,3,(nexps, 5))
    #sel = np.eye(6, 5, dtype=int)
    
    for mode in range(nexps):
        calc = 0
        for i in range(5):
            layers[i].FI_enable=True
            layers[i].selector_args['rate']=0.00008*sel[mode, i]
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

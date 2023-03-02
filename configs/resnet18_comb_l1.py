from random import sample
import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.resnet18 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_activation: true
FI_enable: false
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: 8
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.00001
sub_modules:
  layer1:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
      1:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
  layer2:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
      1:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
observer:
    map: mae
    reduce: sum
'''

perlayer = [0.37398,0.33975,0.13531,0.23271,0.09775,0.27606,0.13421,0.24089]

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
    getattr(FI_network.layer1,'0').conv1,
    getattr(FI_network.layer1,'0').conv2,
    getattr(FI_network.layer1,'1').conv1,
    getattr(FI_network.layer1,'1').conv2,
    getattr(FI_network.layer2,'0').conv1,
    getattr(FI_network.layer2,'0').conv2,
    getattr(FI_network.layer2,'1').conv1,
    getattr(FI_network.layer2,'1').conv2,
    ]

    nexps=32
    np.random.seed(0)
    sel = np.random.randint(0,3,(nexps, len(layers)))
    #sel = np.eye(len(layers)+1, len(layers), dtype=int)
    
    for mode in range(sel.shape[0]):
        calc,calc2 = 0,0
        for i in range(len(layers)):
            layers[i].FI_enable=True
            layers[i].selector_args['rate']=0.00008*sel[mode, i]
            calc+=(perlayer[i])*sel[mode, i]
            calc2+=(perlayer[i]**2)*sel[mode, i]
            layers[i].update_selector()
        calc2=np.sqrt(calc2)
            
        print(''.join(map(str, list(sel[mode]))), end='\t')
        data=iter(testloader)
        FI_network.reset_observe_value()
        '''
        acc=0
        for i in range(total):
            images, labels = next(data)
            images=images.to(device)
            FI_network(images, golden=True)
            out=FI_network(images).cpu().numpy()
            acc+=(np.argmax(out[0])==labels.numpy()[0])

        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f"%(value/total), end='\t')
        '''
        #print("%.5f"%calc, end='\t')
        print("%.5f"%calc2, end='\t')
        print()
        #print("%.2f%%"%(acc/total*100), flush=True)

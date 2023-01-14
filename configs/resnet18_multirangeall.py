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
flip_mode: flip_int_random
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: 8
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
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
  layer3:
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
  layer4:
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
    map: mse
    reduce: sum
'''

def experiment(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    rates=np.logspace(-6, -3, 31)
    '''
    selectedlayers=[
      [0],
      [1,2,3,4],
      [5,6,7,8],
      [9,10,11,12],
      [13,14,15,16],
    ]
    '''
    selectedlayers=[list(range(1,17))]
    layers=[
    # FI_network.conv1,
    getattr(FI_network.layer1,'0').conv1,
    getattr(FI_network.layer1,'0').conv2,
    getattr(FI_network.layer1,'1').conv1,
    getattr(FI_network.layer1,'1').conv2,
    getattr(FI_network.layer2,'0').conv1,
    getattr(FI_network.layer2,'0').conv2,
    getattr(FI_network.layer2,'1').conv1,
    getattr(FI_network.layer2,'1').conv2,
    getattr(FI_network.layer3,'0').conv1,
    getattr(FI_network.layer3,'0').conv2,
    getattr(FI_network.layer3,'1').conv1,
    getattr(FI_network.layer3,'1').conv2,
    getattr(FI_network.layer4,'0').conv1,
    getattr(FI_network.layer4,'0').conv2,
    getattr(FI_network.layer4,'1').conv1,
    getattr(FI_network.layer4,'1').conv2,
    ]

    ranges=np.concatenate([np.arange(2, 4, 0.5), np.arange(4, 8, 1), np.arange(8,24,2)])
    print(ranges)

    for r in ranges:
        data=iter(testloader)
        FI_network.reset_observe_value()
        for layer in layers:
            layer.layerwise_quantization_dynamic_range = r

        print("%2.2f "%r, end=' ')
        acc, acc_g = 0, 0
        for i in range(total):
            images, labels = next(data)
            images=images.to(device)
            out_g = FI_network(images, golden=True).cpu().numpy()
            out=FI_network(images).cpu().numpy()  
            #out=out_g
            acc_g += (np.argmax(out_g[0])==labels.numpy()[0])
            acc+=(np.argmax(out[0])==labels.numpy()[0])

        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f "%np.sqrt(value/total), end='')
        print("%.2f%% %.2f%%"%(acc_g/total*100, acc/total*100), flush=True)
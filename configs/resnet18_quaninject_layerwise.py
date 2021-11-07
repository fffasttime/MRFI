import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.resnet18 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_activation: false
FI_enable: false
FI_weight: true
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: auto
selector: RandomPositionSelector_FixN
selector_args:
  n: 1
sub_modules:
  conv1:
    FI_enable: true
    observer:
      map: mse
      reduce: sum
  layer1:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
      1:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
    observer:
      map: mse
      reduce: sum
  layer2:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
      1:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
    observer:
      map: mse
      reduce: sum
  layer3:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
      1:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
    observer:
      map: mse
      reduce: sum
  layer4:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
      1:
        sub_modules:
          conv1:
            FI_enable: true
          conv2:
            FI_enable: true
          bn1:
            observer:
              map: mse
              reduce: sum
          bn2:
            observer:
              map: mse
              reduce: sum
    observer:
      map: mse
      reduce: sum
observer:
    map: mse
    reduce: sum
'''

def experiment(total = 10000):
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    layers=[
    FI_network.conv1,
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

    for i, inject_layer in enumerate(layers):
        data=iter(testloader)
        FI_network.reset_observe_value()
        for layer in layers:
            layer.FI_enable = False
        inject_layer.FI_enable = True
        print("%2d "%i, end=' ')
        acc=0
        for i in range(total):
            images, labels = next(data)
            FI_network(images, golden=True)
            out=FI_network(images)
            acc+=(np.argmax(out[0])==labels[0])

        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f "%np.sqrt(value/total), end='')
        print("%.2f%%"%(acc/total*100), flush=True)

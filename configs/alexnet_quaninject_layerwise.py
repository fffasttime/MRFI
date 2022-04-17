import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.alexnet import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg_orig='''
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
  features:
    sub_modules:
      0:
        observer:
          map: var
          reduce: sum
      3:
        observer:
          map: var
          reduce: sum
      6:
        observer:
          map: var
          reduce: sum
      8:
        observer:
          map: var
          reduce: sum
      10:
        observer:
          map: var
          reduce: sum
  classifier:
    sub_modules:
      1:
        observer:
          map: var
          reduce: sum
      4:
        observer:
          map: var
          reduce: sum
      6:
        observer:
          map: var
          reduce: sum
observer:
    map: var
    reduce: sum
'''

def exp_orig(total = 10000):
    config = yaml.load(yamlcfg_orig)

    net=Net(pretrained=True)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    acc=0
    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        out=FI_network(images)
        acc+=(np.argmax(out[0])==labels[0])

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, np.sqrt(value/total))
    print("%.2f%%"%(acc/total*100))

yamlcfg='''
FI_activation: false
FI_enable: false
FI_weight: true
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 16
  dynamic_range: auto
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.00001
sub_modules:
  features:
    sub_modules:
      0:
        observer:
          map: mse
          reduce: sum
      3:
        observer:
          map: mse
          reduce: sum
      6:
        observer:
          map: mse
          reduce: sum
      8:
        observer:
          map: mse
          reduce: sum
      10:
        observer:
          map: mse
          reduce: sum
  classifier:
    sub_modules:
      1:
        observer:
          map: mse
          reduce: sum
      4:
        observer:
          map: mse
          reduce: sum
      6:
        observer:
          map: mse
          reduce: sum
observer:
    map: mse
    reduce: sum
'''

def exp(total = 10000):
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    layers=[
    getattr(FI_network.features,'0'),
    getattr(FI_network.features,'3'),
    getattr(FI_network.features,'6'),
    getattr(FI_network.features,'8'),
    getattr(FI_network.features,'10'),
    getattr(FI_network.classifier,'1'),
    getattr(FI_network.classifier,'4'),
    getattr(FI_network.classifier,'6'),
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

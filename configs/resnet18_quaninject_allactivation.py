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
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: 4
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.00001
sub_modules:
  conv1:
    FI_enable: true
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
    config = yaml.load(yamlcfg)

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

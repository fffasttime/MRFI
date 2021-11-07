import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.resnet18 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_enable: false
sub_modules:
  conv1:
    FI_enable: true
    layerwise_quantization:
      bit_width: 8
      dynamic_range: 1
    observer:
      pre_hook: true
      map: maxabs
      reduce: sum
  layer1:
    sub_modules:
      0:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
      1:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
  layer2:
    sub_modules:
      0:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
      1:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
  layer3:
    sub_modules:
      0:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
      1:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
  layer4:
    sub_modules:
      0:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
      1:
        sub_modules:
          conv1:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
          conv2:
            observer:
              pre_hook: true
              map: maxabs
              reduce: sum
observer:
    map: maxabs
    reduce: sum
'''

def experiment(total = 10000):
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    print("max layer")
    data=iter(testloader)
    acc=0
    for i in range(total):
        images, labels = next(data)
        out=FI_network(images)
        acc+=(np.argmax(out[0])==labels[0])

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, value/total)
    print("%.2f%%"%(acc/total*100))

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
    observer:
      pre_hook: true
      map: var
      reduce: sum
  layer1:
    observer:
      pre_hook: true
      map: var
      reduce: sum
  layer2:
    observer:
      pre_hook: true
      map: var
      reduce: sum
  layer3:
    observer:
      pre_hook: true
      map: var
      reduce: sum
  layer4:
    observer:
      pre_hook: true
      map: var
      reduce: sum
observer:
    map: var
    reduce: sum
'''

def experiment(total = 10000):
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    print("4x var layer")
    data=iter(testloader)
    acc=0
    for i in range(total):
        images, labels = next(data)
        out=FI_network(images)
        acc+=(np.argmax(out[0])==labels[0])

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, 4*np.sqrt(value/total))
    print("%.2f%%"%(acc/total*100))

    print("max layer")
    FI_network.reset_observe_value()
    data=iter(testloader)
    for layer in FI_network.subinjectors:
        layer.mapper=mrfi.observer.mapper_maxabs
    
    for i in range(total):
        images, labels = next(data)
        FI_network(images)

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, value/total)

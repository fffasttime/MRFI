import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.alexnet import Net, testset
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
  features:
    sub_modules:
      0:
        FI_enable: true
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
    map: var
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

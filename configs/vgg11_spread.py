import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.vgg11 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg_act='''
FI_activation: true
FI_enable: false
FI_weight: false
flip_mode: set_value
flip_mode_args:
  value: 100 
layerwise_quantization:
  bit_width: 16
  dynamic_range: 64
selector: RandomPositionSelector_FixN
selector_args:
  n: 1
sub_modules:
  features:
    sub_modules:
      0:
        observer:
          map: custom
          reduce: sum
      3:
        observer:
          map: custom
          reduce: sum
      6:
        observer:
          map: custom
          reduce: sum
      8:
        observer:
          map: custom
          reduce: sum
      11:
        observer:
          map: custom
          reduce: sum
      13:
        observer:
          map: custom
          reduce: sum
      16:
        observer:
          map: custom
          reduce: sum
      18:
        observer:
          map: custom
          reduce: sum
  classifier:
    sub_modules:
      0:
        observer:
          map: custom
          reduce: sum
      3:
        observer:
          map: custom
          reduce: sum
      6:
        observer:
          map: custom
          reduce: sum
observer:
    map: custom
    reduce: sum
'''

def exp(total = 10000, weight = False):
    mrfi.observer.Mapper_Dict['custom'] = lambda x, golden: np.array([np.sum(x!=golden), np.sum(x==x), np.mean(x!=golden)])
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg_act)
    if weight:
        config['FI_weight'] = True
        config['FI_activation'] = False

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
    getattr(FI_network.features,'13'),
    getattr(FI_network.features,'16'),
    getattr(FI_network.features,'18'),
    getattr(FI_network.classifier,'0'),
    getattr(FI_network.classifier,'3'),
    getattr(FI_network.classifier,'6'),
    ]

    for i, inject_layer in enumerate(layers[:1]):
        data=iter(testloader)
        FI_network.reset_observe_value()
        for layer in layers:
            layer.FI_enable = False
        inject_layer.FI_enable = True
        print("%2d "%i, end=' ')
        acc=0
        for i in range(total):
            images, labels = next(data)
            images=images.to(device)
            FI_network(images, golden=True)
            out=FI_network(images).cpu().numpy()

        observes=FI_network.get_observes()
        for name, value in observes.items():
            value = value / total
            print(name, "%6.2f/%.0f, %.2f%%"%(value[0],value[1], value[2]*100))

from numpy.lib.type_check import imag
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
observer:
  map: mse
  reduce: sum
selector: RandomPositionSelector_FixN
selector_args:
  n: 1
sub_modules:
  conv1:
    FI_enable: true
    layerwise_quantization:
      bit_width: 8
      dynamic_range: 1
    observer:
      map: mse
      reduce: sum
  layer1:
    observer:
      map: mse
      reduce: sum
  layer2:
    observer:
      map: mse
      reduce: sum
  layer3:
    observer:
      map: mse
      reduce: sum
  layer4:
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    acc=0
    for i in range(total):
        images, labels = next(data)
        images=images.to(device)
        FI_network(images, golden=True)
        out=FI_network(images).cpu().numpy()
        acc+=(np.argmax(out[0])==labels[0].numpy())

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, np.sqrt(value/total))
    print("%.2f%%"%(acc/total*100))

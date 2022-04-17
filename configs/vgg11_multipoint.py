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
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
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
          map: mse
          reduce: sum
      3:
        FI_enable: true
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
      11:
        observer:
          map: mse
          reduce: sum
      13:
        observer:
          map: mse
          reduce: sum
      16:
        observer:
          map: mse
          reduce: sum
      18:
        observer:
          map: mse
          reduce: sum
  classifier:
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
observer:
    map: mse
    reduce: sum
'''

def exp_act(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg_act)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    injects=[1, 2, 4, 8, 16]

    for ninject in injects:
        data=iter(testloader)
        FI_network.reset_observe_value()
        layer = getattr(FI_network.features,'3')
        layer.selector_args={'n': ninject}
        layer.update_selector()

        print("%2d "%ninject, end=' ')
        acc=0
        for i in range(total):
            images, labels = next(data)
            images=images.to(device)
            FI_network(images, golden=True)
            out=FI_network(images).cpu().numpy()  
            acc+=(np.argmax(out[0])==labels.numpy()[0])

        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f "%np.sqrt(value/total), end='')
        print("%.2f%%"%(acc/total*100), flush=True)

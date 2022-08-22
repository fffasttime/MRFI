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
  bit_width: 6
layerwise_quantization:
  bit_width: 6
  dynamic_range: 64
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
  poisson: True
sub_modules:
  features:
    sub_modules:
      3:
        FI_enable: true
      6:
        FI_enable: true
      8:
        FI_enable: true
      11:
        FI_enable: true
      13:
        FI_enable: true
      16:
        FI_enable: true
      18:
        FI_enable: true
observer:
    map: mse
    reduce: sum
'''

def exp(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg_act)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    ranges=np.concatenate([np.arange(2, 4, 0.5), np.arange(4, 8, 1), np.arange(8,64,4)])
    print(ranges)

    for r in ranges:
        data=iter(testloader)
        FI_network.reset_observe_value()
        lnum = [3,6,8]
        for l in lnum:
            layer = getattr(FI_network.features,str(l))
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

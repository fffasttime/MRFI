import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.lenet_cifar import Net, testset
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
  dynamic_range: 8
selector: FixPositionSelector
selector_args:
  pos: []
sub_modules:
  conv1:
    FI_enable: true
  fc3:
    observer:
      map: mse
      reduce: sum
'''

def experiment(total = 10000):
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True)
    device = torch.device("cpu") # lenet uses cpu only
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    for i in range(3072):
        FI_network.conv1.selector_args['pos'] = [i]
        FI_network.conv1.update_selector()

        data=iter(testloader)
        FI_network.reset_observe_value()
        
        acc=0
        for i in range(total):
            images, labels = next(data)
            images=images.to(device)
            outg = FI_network(images, golden=True)
            out=FI_network(images).cpu().numpy()
            acc+=(np.argmax(out[0])==labels.numpy()[0])

        observes=FI_network.get_observes()
        for name, value in observes.items():
            print("%.5f "%np.sqrt(value/total), end='')
        print("%.2f"%(acc/total*100), flush=True)
        

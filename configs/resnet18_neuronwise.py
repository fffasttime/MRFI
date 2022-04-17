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
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: auto
  dynamic_range_weight: 2
selector: FixPositionSelector
selector_args:
  pos: []
sub_modules:
  layer1:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_enable: true
observer:
  map: mse
  reduce: sum
'''

def experiment(total = 100):
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    FI_network = ModuleInjector(net, config)
    layerFI = getattr(FI_network.layer1,'0').conv1

    for i in range(36864):
        layerFI.selector_args['pos'] = [i]
        layerFI.update_selector()

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

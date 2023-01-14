import yaml
import logging
import torch
import numpy as np
import time

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
  dynamic_range: 8
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.00001
sub_modules:
  conv1:
    FI_enable: false
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
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    batch_size = 50

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    FI_network = ModuleInjector(net, config)

    rates=np.logspace(-6, -5, 11)

    selectedlayers=[
      []
    ]

    layers=[
    FI_network.conv1,
    getattr(FI_network.layer1,'0').conv1,
    getattr(FI_network.layer1,'0').conv2,
    getattr(FI_network.layer1,'1').conv1,
    getattr(FI_network.layer1,'1').conv2,
    getattr(FI_network.layer2,'0').conv1,
    getattr(FI_network.layer2,'0').conv2,
    getattr(FI_network.layer2,'1').conv1,
    getattr(FI_network.layer2,'1').conv2,
    getattr(FI_network.layer3,'0').conv1,
    getattr(FI_network.layer3,'0').conv2,
    getattr(FI_network.layer3,'1').conv1,
    getattr(FI_network.layer3,'1').conv2,
    getattr(FI_network.layer4,'0').conv1,
    getattr(FI_network.layer4,'0').conv2,
    getattr(FI_network.layer4,'1').conv1,
    getattr(FI_network.layer4,'1').conv2,
    ]

    for mode in [mrfi.flip_mode.flip_int_highest, mrfi.flip_mode.flip_int_random]:
        T0 = time.time()
        for ri, rate in enumerate(rates):
            data=iter(testloader)

            for ri, layer in enumerate(layers):
                layer.selector_args['rate']=rate
                layer.selector_args['poisson']=True
                layer.update_selector()
                layer.flip_mode = mode

            FI_network.reset_observe_value()

            acc=0
            for i in range(total//batch_size):
                images, labels = next(data)
                images=images.to(device)
                out_free=FI_network(images, golden=True).cpu().numpy()
                out=FI_network(images).cpu().numpy()
                acc+=np.count_nonzero(np.argmax(out, axis=1)==labels.numpy())

            observes=FI_network.get_observes()

            print('%.10f'%rate, end='\t')
            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), end='\t')
            print("%.2f%%"%(acc/total*100), flush=True)
            
        print('done in %.2f s'%(time.time()-T0))
        rates*=np.sqrt(6) ##### larger

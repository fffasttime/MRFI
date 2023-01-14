import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.resnet18 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_activation: true
FI_enable: false
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: 8
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.00001
sub_modules:
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

def exp(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    rates=np.logspace(-6, -3, 31)

    layers=[
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

    for mode in [mrfi.flip_mode.flip_int_highest]:

        data=iter(testloader)
        images, labels = next(data) # single picture 0
        images, labels = next(data) # single picture 1
        images, labels = next(data) # single picture 2
        images, labels = next(data) # single picture 3
        images, labels = next(data) # single picture 4
        images, labels = next(data) # single picture 5
        images, labels = next(data) # single picture 6
        images, labels = next(data) # single picture 7
        images=images.to(device)

        out_free=FI_network(images, golden=True).cpu().numpy()

        print(np.std(out_free), np.argmax(out_free[0])==labels.numpy()[0])
        # print(np.sort(out_free[0]))

        for ri, rate in enumerate(rates):

            for layer in layers:
                layer.selector_args['rate']=rate
                layer.selector_args['poisson']=True
                layer.update_selector()
                layer.flip_mode = mode

            FI_network.reset_observe_value()

            acc=0
            for i in range(total):
                out=FI_network(images).cpu().numpy()
                acc+=(np.argmax(out[0])==labels.numpy()[0])
            
            observes=FI_network.get_observes()

            print('%.10f'%rate, end='\t')
            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), end='\t')
            print('%.2f%%'%(acc/total*100), flush=True)

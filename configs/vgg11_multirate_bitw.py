import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.vgg11 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg1='''
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: 48
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
sub_modules:
  features:
    sub_modules:
      3:
      6:
      8:
      11:
      13:
      16:
      18:
observer:
    map: mse
    reduce: sum
'''

yamlcfg2='''
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: 48
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
sub_modules:
  features:
    sub_modules:
      3:
      6:
      8:
      11:
      13:
      16:
      18:
observer:
    map: mse
    reduce: sum
'''

def exp_cfg(cfg, total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(cfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    rates=np.logspace(-5, -3, 21)


    for mode in [mrfi.flip_mode.flip_int_random]:

        golden_class = np.ndarray((len(rates), total), np.int)
        inject_class = np.ndarray((len(rates), total), np.int)
        label_class = np.ndarray((len(rates), total), np.int)

        for ri, rate in enumerate(rates):
            data=iter(testloader)

            for layer in FI_network.features.subinjectors:
                layer.selector_args['rate']=rate
                layer.selector_args['poisson']=True
                layer.update_selector()
                layer.flip_mode = mode

            FI_network.reset_observe_value()

            acc=0
            for i in range(total):
                images, labels = next(data)
                images=images.to(device)
                out_free=FI_network(images, golden=True).cpu().numpy()
                out=FI_network(images).cpu().numpy()
                acc+=(np.argmax(out[0])==labels.numpy()[0])

                golden_class[ri, i] = np.argmax(out_free[0])
                inject_class[ri, i] = np.argmax(out[0])
                label_class[ri, i] = labels.numpy()[0]
            
            observes=FI_network.get_observes()

            print('%.10f'%rate, end='\t')
            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), end='\t')
            print('%.2f%%'%(acc/total*100), flush=True)

            # np.save('vgg11_multirate_classout.npy', [golden_class, inject_class, label_class])

def exp(total = 10000):
    exp_cfg(yamlcfg1, total)
    exp_cfg(yamlcfg2, total)
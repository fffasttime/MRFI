import yaml
import operator
from functools import reduce
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import mrfi.observer
import mrfi.selector
from model.resnet18 import Net, testset
from mrfi.flip_mode import flip_int_highest
from mrfi.injector import ModuleInjector
from torch import autograd

yamlcfg='''
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: 8
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def experiment(total = 1000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    data=iter(testloader)

    n_confidence = np.zeros(total)
    n_confidence2 = np.zeros(total)
    n_tconfidence = np.zeros(total)
    r_acc = np.zeros(total, dtype=np.bool)
    fi_acc = np.zeros(total, dtype=np.bool)
    fi_acc_dmr = np.zeros(total, dtype=np.bool)
    fi_acc_tmr = np.zeros(total, dtype=np.bool)

    FI_network.reset_observe_value()
    acc, acc0, acc_dmr, acc_tmr=0,0,0,0
    for i in range(total):
        images, labels = next(data)
        images=images.to(device)
        out_free=FI_network(images, golden=True)
        out=FI_network(images)
        out2=FI_network(images)
        out3=FI_network(images)
        n_confidence[i] = 1-np.max(out[0].softmax(0).cpu().numpy())
        n_confidence2[i] = 1-np.max(out2[0].softmax(0).cpu().numpy())
        n_tconfidence[i] = 1-np.max(out_free[0].softmax(0).cpu().numpy())

        out_free = out_free.cpu().numpy()
        out = out.cpu().numpy()
        out2 = out2.cpu().numpy()
        out3 = out3.cpu().numpy()

        fi_class = np.argmax(out[0])
        fi_class2 = np.argmax(out2[0])
        fi_class3 = np.argmax(out3[0])
        fi_class_dmr = fi_class if n_confidence[i]<n_confidence2[i] else fi_class2 # DMR class
        free_class = np.argmax(out_free[0])
        fi_class_tmr = fi_class if fi_class==fi_class2 or fi_class==fi_class3 else fi_class2
        #print(labels.numpy()[0], free_class, fi_class, fi_class2, fi_class3)

        acc+=(fi_class==labels.numpy()[0])
        acc_dmr+=(fi_class_dmr==labels.numpy()[0])
        acc_tmr+=(fi_class_tmr==labels.numpy()[0])
        acc0+=(free_class==labels.numpy()[0])
        r_acc[i] = (free_class==labels.numpy()[0])
        fi_acc[i] = (fi_class==labels.numpy()[0])
        fi_acc_dmr[i] = (fi_class_dmr==labels.numpy()[0])
        fi_acc_tmr[i] = (fi_class_tmr==labels.numpy()[0])

    observes=FI_network.get_observes()

    for name, value in observes.items():
        print("%.4f"%np.sqrt(value/total), end='\t')
    print("%.2f%%"%(acc0/total*100), end='\t')
    print("%.2f%%"%(acc_dmr/total*100), end='\t')
    print("%.2f%%"%(acc_tmr/total*100), end='\t')
    print("%.2f%%"%(acc/total*100), flush=True)

    np.savez('_logs/dyn_allact.npz', nconf=n_confidence, ntconf = n_tconfidence, 
             racc=r_acc, fiacc=fi_acc, nconf2=n_confidence2, fiacc_dmr = fi_acc_dmr, fiacc_tmr=fi_acc_tmr)
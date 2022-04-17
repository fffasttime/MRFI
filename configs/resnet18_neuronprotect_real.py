import yaml
import operator
from functools import reduce
import torch
import numpy as np

import mrfi.observer
import mrfi.selector
from model.resnet18 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: 4
  dynamic_range_weight: 4
selector: RandomPositionSelector_FixN
selector_args:
  n: 0
sub_modules:
  layer1:
    sub_modules:
      0:
        sub_modules:
          conv1:
            FI_weight: true
            FI_activation: false
            selector_args:
              n: 10
          conv2:
      1:
        sub_modules:
          conv1:
          conv2:
  layer2:
    sub_modules:
      0:
        sub_modules:
          conv1:
          conv2:
      1:
        sub_modules:
          conv1:
          conv2:
  layer3:
    sub_modules:
      0:
        sub_modules:
          conv1:
          conv2:
      1:
        sub_modules:
          conv1:
          conv2:
  layer4:
    sub_modules:
      0:
        sub_modules:
          conv1:
          conv2:
      1:
        sub_modules:
          conv1:
          conv2:
observer:
  map: mse
  reduce: sum
'''

''' select 1 pos randomly, ignore pos'''
class CustomSelector:
    def __init__(self, pos, n):
        self.pos = sorted(pos)
        self.n = n
    
    def gen_list(self, shape):
        if len(shape) > 1:
            shape=reduce(operator.mul, shape)
        ret = []
        for _ in range(self.n):
          p = np.random.randint(0, shape)
          if p not in self.pos:
              ret.append(p)
        return ret

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_protect(FI_network, testloader, total, show_golden=False):
    data=iter(testloader)
    FI_network.reset_observe_value()
    
    acc=0
    accg=0
    for i in range(total):
        images, labels = next(data)
        images=images.to(device)
        outg = FI_network(images, golden=True).cpu().numpy()
        out=FI_network(images).cpu().numpy()
        acc+=(np.argmax(out[0])==labels.numpy()[0])
        accg+= (np.argmax(outg[0])==labels.numpy()[0])

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print("%.5f"%np.sqrt(value/total), end='\t')
    print("%.2f%%"%(acc/total*100), flush=True)

    if show_golden:
        print("golden")
        print("%.2f%%"%(accg/total*100), flush=True)

def experiment(total = 10000):
    #prot10 = 3686
    #nprots = [prot10, prot10*2, prot10*3, prot10*5]
    #stat = np.loadtxt('_logs/resnet18_neuronwise.txt', skiprows=1)
    #grad = np.loadtxt('_logs/resnet18_grad6.txt')

    mrfi.selector.Selector_Dict['custom']=CustomSelector
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True)
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    FI_network = ModuleInjector(net, config)

    layerFI = getattr(FI_network.layer1,'0').conv1

    print('no')
    run_protect(FI_network, testloader, total, True)

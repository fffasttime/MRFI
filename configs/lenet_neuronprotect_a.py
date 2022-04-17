import yaml
import operator
from functools import reduce
import torch
import numpy as np

import mrfi.observer
import mrfi.selector
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
selector: custom
selector_args:
  pos: []
  n: 5
sub_modules:
  conv2:
    FI_enable: true
  fc3:
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

def experiment(total = 10000):
    nprots = [118, 235, 353, 588]
    stat = np.loadtxt('neuronwise100.txt', skiprows=1)
    grad = np.loadtxt('grad2.txt')

    mrfi.selector.Selector_Dict['custom']=CustomSelector
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True)
    device = torch.device("cpu") # lenet uses cpu only
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)
    
    for nprot in nprots:
        prot_policy={
        'random' : np.argsort(np.random.rand(1176))[:nprot],
        #'acc' : np.argsort(stat[:, 1])[:nprot],
        #'rmse' : np.argsort(stat[:, 0])[::-1][:nprot],
        #'abs' : np.argsort(grad[:,0])[::-1][:nprot],
        #'gradabs' : np.argsort(grad[:,1])[::-1][:nprot],
        #'hessian' : np.argsort(grad[:,2])[::-1][:nprot],
        }
        print(nprot)
        for name, prot in prot_policy.items():
            print(name, end='\t')
            FI_network.conv2.selector_args['pos'] = prot
            FI_network.conv2.update_selector()

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
                print("%.5f"%np.sqrt(value/total), end='\t')
            print("%.2f%%"%(acc/total*100), flush=True)
            

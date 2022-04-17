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
FI_activation: false
FI_enable: true
FI_weight: true
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: auto
  dynamic_range_weight: 2
selector: custom
selector_args:
  pos: []
  axis: 0
  n: 10
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

''' select 1 channel randomly, ignore pos'''
class CustomSelector:
    def __init__(self, pos, axis, n):
        self.pos = sorted(pos)
        self.axis = axis
        self.n = n
    
    def gen_list(self, shape):
        if len(shape) > 1:
            shape=reduce(operator.mul, shape)
        ret = []
        for _ in range(self.n):
          p = np.random.randint(0, shape)
          if self.axis==0:
              if p//(64*3*3) not in self.pos:
                  ret.append(p)
          else:
              if (p//9%64) not in self.pos:
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
    prot10 = 3686
    nprots = [6, 12, 18, 24, 30]
    stat = np.loadtxt('_logs/resnet18_neuronwise.txt', skiprows=1)
    grad = np.loadtxt('_logs/resnet18_grad6.txt')

    mrfi.selector.Selector_Dict['custom']=CustomSelector
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True)
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    FI_network = ModuleInjector(net, config)

    layerFI = getattr(FI_network.layer1,'0').conv1

    rmse_channel_out = np.sqrt(np.sum(stat[:, 0].reshape(64, 64, 9)**2, axis=(0,2)))
    gradabs_channel_out = np.sum(grad[:, 3].reshape(64, 64, 9), axis=(0,2))
    rmse_channel_in = np.sqrt(np.sum(stat[:, 0].reshape(64, 64, 9)**2, axis=(1,2)))
    gradabs_channel_in = np.sum(grad[:, 3].reshape(64, 64, 9), axis=(1,2))
    print(rmse_channel_in)
    print(rmse_channel_out)

    #print('no')
    #run_protect(FI_network, testloader, total, True)
    
    for nprot in nprots:
        prot_policy={
        'random, out' : np.argsort(np.random.rand(64))[:nprot],
        #'acc' : np.argsort(stat[:, 1])[:nprot],
        'rmse, in' : np.argsort(rmse_channel_in)[::-1][:nprot],
        'rmse, out' : np.argsort(rmse_channel_out)[::-1][:nprot],
        #'abs' : np.argsort(grad[:,0])[::-1][:nprot],
        #'gradabs' : np.argsort(grad[:,1])[::-1][:nprot],
        #'hessian' : np.argsort(grad[:,2])[::-1][:nprot],
        'gradabs*delta, in' : np.argsort(gradabs_channel_in)[::-1][:nprot],
        'gradabs*delta, out' : np.argsort(gradabs_channel_out)[::-1][:nprot],
        #'hessian*delta' : np.argsort(grad[:,4])[::-1][:nprot],
        #'delta' : np.argsort(grad[:,5])[::-1][:nprot],
        }
        print(nprot)
        for name, prot in prot_policy.items():
            print(name, end='\t')
            layerFI.selector_args['pos'] = prot
            if 'in' in name:
                layerFI.selector_args['axis'] = 0
            else:
                layerFI.selector_args['axis'] = 1
            layerFI.update_selector()

            run_protect(FI_network, testloader, total)


            

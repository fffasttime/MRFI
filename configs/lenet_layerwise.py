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
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0008
sub_modules:
  conv1:
    observer:
      map: mse
      reduce: sum
  conv2:
    observer:
      map: mse
      reduce: sum
  fc1:
    observer:
      map: mse
      reduce: sum
  fc2:
    observer:
      map: mse
      reduce: sum
  fc3:
    observer:
      map: mse
      reduce: sum
'''

def experiment(total = 10000):
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    for layer in FI_network.subinjectors:
      data=iter(testloader)
      FI_network.reset_observe_value()
      
      accg=0
      acc=0
      for ll in FI_network.subinjectors:
        ll.FI_enable = False
      layer.FI_enable = True
      for i in range(total):
          images, labels = next(data)
          images=images.to(device)
          outg = FI_network(images, golden=True)
          out=FI_network(images).cpu().numpy()
          accg+=(np.argmax(outg[0])==labels.numpy()[0])
          acc+=(np.argmax(out[0])==labels.numpy()[0])

      observes=FI_network.get_observes()
      print(layer.name, end='\t')
      for name, value in observes.items():
          print("%.5f"%np.sqrt(value/total), end='\t')
      print("%.2f%%"%(acc/total*100), flush=True, end='\t')
      print("%.2f%%"%(accg/total*100), flush=True)

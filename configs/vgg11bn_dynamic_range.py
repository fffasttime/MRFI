import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.vgg11bn import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_enable: false
sub_modules:
  features:
    sub_modules:
      0:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      4:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      8:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      11:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      15:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      18:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      22:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      25:
        observer:
          pre_hook: true
          map: var
          reduce: sum
  classifier:
    sub_modules:
      0:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      3:
        observer:
          pre_hook: true
          map: var
          reduce: sum
      6:
        observer:
          pre_hook: true
          map: var
          reduce: sum
observer:
    map: var
    reduce: sum
'''

def experiment(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    print("4x var layer")
    data=iter(testloader)
    acc=0
    for i in range(total):
        images, labels = next(data)
        images = images.to(device)
        out=FI_network(images).cpu().numpy()
        acc+=(np.argmax(out[0])==labels.numpy()[0])

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, 4*np.sqrt(value/total))
    print("%.2f%%"%(acc/total*100))

    print("max layer")
    FI_network.reset_observe_value()
    data=iter(testloader)
    for layer in FI_network.features.subinjectors:
        layer.mapper=mrfi.observer.mapper_maxabs
    for layer in FI_network.classifier.subinjectors:
        layer.mapper=mrfi.observer.mapper_maxabs
    
    for i in range(total):
        images, labels = next(data)
        images = images.to(device)
        FI_network(images)

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, value/total)

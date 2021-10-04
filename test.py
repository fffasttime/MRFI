from operator import mod

from torch.nn.modules.module import T
from configs.configurator import Configurator
from model.cnn_exp import Net, testset
from mrfi.injector import ModuleInjector
import mrfi.observer
import matplotlib.pyplot as plt

import yaml
import logging
import torch
import numpy as np

#logging.basicConfig(level=logging.NOTSET)

#configurator = Configurator(Net(), ['conv'])
#configurator.save('cnn_exp.yaml')

def default_inject():
    with open("configs/cnn_exp.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = 10000

    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    print(FI_network.observe_value)
    print(np.sqrt(FI_network.observe_value/total))

def get_dynamic_range():
    with open("configs/cnn_exp_dynamic_range.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    FI_network = ModuleInjector(net, config)
    total = 10000

    for i in range(total):
        images, labels = next(data)
        FI_network(images)

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, 4*np.sqrt(value/total))

def quan_inject():
    with open("configs/cnn_exp_quan.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = 10000
    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, np.sqrt(value/total))

def coverage_test():
    mrfi.observer.Mapper_Dict['custom'] = lambda x, golden: np.array([np.sum(x!=golden), np.sum(x==x), np.mean(x!=golden)])

    with open("configs/cnn_exp_coverage.yaml") as f:
        config = yaml.full_load(f)
    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()
    FI_network = ModuleInjector(net, config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = 10000
    
    for module in FI_network.subinjectors:
        if 'conv' in module.name:
            module.FI_activation = True
    print('Activation')

    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)
    
    observes=FI_network.get_observes()
    for name, value in observes.items():
        value = value / total
        print(name, "%6.2f/%.0f, %.2f%%"%(value[0],value[1], value[2]*100))
    

    data=iter(testloader)
    FI_network.reset_observe_value()
    for module in FI_network.subinjectors:
        if 'conv' in module.name:
            module.FI_activation = False
            module.FI_weight = True
    print('Weight')

    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    observes=FI_network.get_observes()
    for name, value in observes.items():
        value = value / total
        print(name, "%6.2f/%0.0f, %.2f%%"%(value[0],value[1], value[2]*100))

def plot_transmission():
    with open("configs/cnn_exp_plot_transmission.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = 200
    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    observes=FI_network.get_observes()
    data=[np.sqrt(x) for x in observes.values()]
    data[5]/=1.34
    data[6]/=0.62
    data[7]/=6.10
    for layer in data:
        print(np.mean(layer))
    plt.boxplot(data,showmeans=True, vert=False, showfliers=False)
    plt.show()

plot_transmission()

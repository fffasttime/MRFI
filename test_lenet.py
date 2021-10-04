from torch.nn.modules.module import T
from configs.configurator import Configurator
from model.lenet_cifar import Net, testset
from mrfi.injector import ModuleInjector
import mrfi.observer

import yaml
import logging
import torch
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats

def lenet_default():
    logging.basicConfig(level=logging.NOTSET)

    with open("configs/lenet_cifar.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar10.pth'))
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

    with open("configs/lenet_cifar.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar10.pth'))
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
        value = value / total
        print(name, "%6.2f/%.0f, %.2f%%"%(value[0],value[1], value[2]*100))

def errorplot():
    '''
    inject on conv2 activation
    map: sumdiff/var
    reduce: no_reduce
    '''
    with open("configs/lenet_cifar_errorplot.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar10.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    first_neuron_diff = lambda x, golden: x.reshape(-1)[0]-golden.reshape(-1)[0]

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = 1000

    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    observes=FI_network.get_observes()
    i=0
    for name, value in observes.items():
        i+=1
        plt.subplot(2,2,i)
        plt.hist(value,31)
        mean=np.mean(value)
        std=np.std(value)
        min,max=np.min(value),np.max(value)
        cx=np.linspace(min,max,100)
        cy=(stats.norm.pdf((cx-mean)/std))/std*(total/31*(max-min))
        plt.plot(cx,cy)

        #value-=np.mean(value)
        #value/=np.std(value)
        #print(np.min(value), np.max(value))
        #stats.probplot(value, plot=plt)
        #print(stats.kstest(value, 'norm'))
        #print(stats.shapiro(value))
        #print(stats.normaltest(value))
        plt.title(name)
    plt.show()

lenet_default()

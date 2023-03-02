from numpy.core.fromnumeric import size
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

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', type=str, default='wa', help='wa/coverage/error')
parser.add_argument('-w', '--weight', default='_data/cifar10.pth', help='weight file path')
parser.add_argument('-n', '--number', type=int, default=10000, help='number of image to test')

opt = parser.parse_args()

def lenet_default():
    logging.basicConfig(level=logging.NOTSET)

    with open("configs/lenet_cifar.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load(opt.weight))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = opt.number

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
    net.load_state_dict(torch.load(opt.weight))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = opt.number

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
    map: sumdiff/var/first
    reduce: no_reduce
    '''
    with open("configs/lenet_cifar_errorplot.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load(opt.weight))
    net.eval()

    first_neuron_diff = lambda x, golden: x.reshape(-1)[0]-golden.reshape(-1)[0]
    mrfi.observer.Mapper_Dict['custom'] = first_neuron_diff

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = opt.number

    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    # # use identity map
    # np.save("lenet_outerror_10000x10.npy",FI_network.fc3.observe_value)
    # return 
    
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
        plt.title(name.split('.')[1])
    plt.show()

def AIC(data, dist, dargs = None, k = None):
    args = dist.fit(data)
    k = len(args) if k is None else k
    if dargs is None:
        logl = dist.logpdf(data, *args).sum()
    else:
        logl = dist.logpdf(data, *dargs).sum()
    return -2*logl + 2 * k, -2*logl, args

def fittest(data):
    #dists = [stats.norm, stats.gamma, stats.lognorm, stats.t, stats.expon, stats.uniform, stats.beta]
    dists = [stats.norm, stats.t]
    for dist in dists:
        print(str(type(dist)).split('.')[-1], *AIC(data, dist))

def WandAplot():
    mrfi.observer.Mapper_Dict['custom'] = lambda x, golden: np.random.choice(x.reshape(-1), 10)

    with open("configs/lenet_cifar_custom.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load(opt.weight))
    net.eval()

    FI_network = ModuleInjector(net, config)

    total = opt.number
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    for i in range(total):
        images, labels = next(data)
        FI_network(images, True)
        FI_network(images)

    observes=FI_network.get_observes()
    
    plt.xticks(size=5)
    i=0
    for layer in ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']:
        i+=1
        plt.subplot(3,3,i)
        value=getattr(FI_network.module, layer).weight.data.numpy().reshape(-1)
        plt.hist(value,31)
        print(layer)
        
        vs = np.random.choice(value, 100)
        fittest(vs)
        desc_stats = stats.describe(value)
        # print(desc_stats)
        
        mean=np.mean(value)
        std=np.std(value)
        min,max=np.min(value),np.max(value)
        cx=np.linspace(min,max,100)
        cy=(stats.norm.pdf((cx-mean)/std))/std*(value.size/31*(max-min))
        plt.plot(cx,cy,color='orange')
        plt.title(layer+'.weight', fontdict={'size':10})
        plt.yticks(size=8)
        plt.xticks(size=8)

    for name, value in observes.items():
        i+=1
        plt.subplot(3,3,i)
        plt.hist(value,31,color='green')
        print(name)

        vs = np.random.choice(value, 100)
        fittest(vs)
        desc_stats = stats.describe(value)
        # print(desc_stats)

        mean=np.mean(value)
        std=np.std(value)
        min,max=np.min(value),np.max(value)
        cx=np.linspace(min,max,100)
        cy=(stats.norm.pdf((cx-mean)/std))/std*(total/31*(max-min)*10)
        plt.plot(cx,cy,color='orange')
        plt.title(name+'.activation', fontdict={'size':10})
        plt.yticks(size=8)
        plt.xticks(size=8)
    plt.subplots_adjust(None, None, None, None, 0.4, 0.55)
    plt.show()

if opt.mode == 'wa':
    WandAplot()
elif opt.mode == 'coverage':
    coverage_test()
elif opt.mode == 'error':
    errorplot()
else:
    raise NotImplementedError

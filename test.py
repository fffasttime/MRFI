from operator import mod

from torch.nn.modules.module import T
from configs.configurator import Configurator
from model.cnn_exp import Net, testset
from mrfi.injector import ModuleInjector, layerwise_dequantization
import mrfi.observer
import mrfi.selector
import mrfi.flip_mode
import matplotlib.pyplot as plt

import yaml
import logging
import torch
import numpy as np
from scipy import stats

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

    FI_network = ModuleInjector(net, config)
    total = 10000

    print("4x var layer")
    data=iter(testloader)
    acc=0
    for i in range(total):
        images, labels = next(data)
        out=FI_network(images)
        acc+=(np.argmax(out[0])==labels[0])

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, 4*np.sqrt(value/total))
    print("%.2f%%"%(acc/total*100))

    print("max layer")
    FI_network.reset_observe_value()
    data=iter(testloader)
    for layer in FI_network.subinjectors:
        layer.mapper=mrfi.observer.mapper_maxabs
    
    for i in range(total):
        images, labels = next(data)
        FI_network(images)

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, value/total)



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
    
    np.save('plot_transmission.npy', data)

def multilayertest():
    with open("configs/cnn_exp_quan.yaml") as f:
        config = yaml.load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    total = 10000

    for mode in range(0,2):
        if mode==0:
            selector = mrfi.selector.RandomPositionSelector_FixN
            selector_args = {'n': 1}
        else:
            selector = mrfi.selector.RandomPositionSelector_Rate
            selector_args = {'rate':1e-3}

        for i in range(1,7):
            data=iter(testloader)
            layer=getattr(FI_network,"conv"+str(i))
            layer.FI_enable=True
            layer.selector=selector(**selector_args)

            FI_network.reset_observe_value()

            acc=0
            for i in range(total):
                images, labels = next(data)
                FI_network(images, golden=True)
                out=FI_network(images)
                acc+=(np.argmax(out[0])==labels[0])

            observes=FI_network.get_observes()

            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), end='\t')
            print('%.2f%%'%(acc/total*100), flush=True)
            layer.FI_enable=False
        print()

def quan_test():
    '''test different activation bound effect, golden run should close quan'''
    
    with open("configs/cnn_exp_quan.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    config['selector']='RandomPositionSelector_FixN'
    config['selector_args']['n']=0
    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    total = 10000

    for i in range(1,8):
        data=iter(testloader)
        if i==7:
            for j in range(1,7):
                layer=getattr(FI_network,"conv"+str(j))
                layer.FI_enable=True
        else:
            layer=getattr(FI_network,"conv"+str(i))
            layer.FI_enable=True

        FI_network.reset_observe_value()

        acc=0
        for i in range(total):
            images, labels = next(data)
            FI_network(images, golden=True)
            out=FI_network(images)
            acc+=(np.argmax(out[0])==labels[0])

        observes=FI_network.get_observes()

        for name, value in observes.items():
            print("%.4f"%np.sqrt(value/total), end='\t')
        print('%.2f%%'%(acc/total*100))
        layer.FI_enable=False
    print()

def multirate_test():
    with open("configs/cnn_exp_multirate.yaml") as f:
        config = yaml.load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    total = 10000

    rates=np.logspace(-6, -2, 41)

    for mode in [mrfi.flip_mode.flip_int_highest, mrfi.flip_mode.flip_int_random]:
        for rate in rates:
            data=iter(testloader)

            for layer in FI_network.subinjectors:
                layer.selector_args['rate']=rate
                layer.selector_args['poisson']=True
                layer.update_selector()
                layer.flip_mode = mode

            FI_network.reset_observe_value()

            acc=0
            for i in range(total):
                images, labels = next(data)
                FI_network(images, golden=True)
                out=FI_network(images)
                acc+=(np.argmax(out[0])==labels[0])
                #print(out)

            observes=FI_network.get_observes()

            print('%.10f'%rate, end='\t')
            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), end='\t')
            print('%.2f%%'%(acc/total*100), flush=True)

def WandAplot():
    mrfi.observer.Mapper_Dict['custom'] = lambda x, golden: np.random.choice((x-golden).reshape(-1)[(x-golden).reshape(-1)!=0], 10)

    with open("configs/cnn_exp_errorplot.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    total = 1000
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    observes=FI_network.get_observes()
    i=0
    for name, value in observes.items():
        i+=1
        plt.subplot(3,3,i)
        plt.hist(value,31,color='green')

        mean=np.mean(value)
        std=np.std(value)
        min,max=np.min(value),np.max(value)
        cx=np.linspace(min,max,100)
        cy=(stats.norm.pdf((cx-mean)/std))/std*(total/31*(max-min)*10)
        plt.plot(cx,cy,color='orange')
        plt.title(name+'.nozeroerr', fontdict={'size':10})
        plt.yticks(size=8)
        plt.xticks(size=8)
    plt.subplots_adjust(None, None, None, None, 0.4, 0.55)
    plt.show()

def exp_coverage():
    with open("configs/cnn_exp_multirate.yaml") as f:
        config = yaml.load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    acc, acc_golden=0, 0
    for _ in range(5):
        data=iter(testloader)
        for i in range(10000):
            images, labels = next(data)
            out_golden=FI_network(images, golden=True)
            out=FI_network(images)
            acc+=(np.argmax(out[0])==labels[0])
            acc_golden+=(np.argmax(out_golden[0])==labels[0])
            print(acc.numpy(), acc_golden.numpy(), FI_network.observe_value, flush=True)

def exp_combination():
    with open("configs/cnn_exp_combination.yaml") as f:
        config = yaml.load(f)
    
    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    total = 10000

    for mode in range(64):
        for i in range(1,7):
            layer=getattr(FI_network,"conv"+str(i))
            layer.FI_enable=bool(mode>>(i-1) & 1)

        FI_network.reset_observe_value()
        data=iter(testloader)
        acc=0
        for i in range(total):
            images, labels = next(data)
            FI_network(images, golden=True)
            out=FI_network(images)
            acc+=(np.argmax(out[0])==labels[0])
        observes=FI_network.get_observes()

        print("%2d"%mode, end='\t')
        for name, value in observes.items():
            print("%.4f"%np.sqrt(value/total), end='\t')
        print('%.2f%%'%(acc/total*100), flush=True)

exp_combination()

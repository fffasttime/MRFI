from configs.configurator import Configurator
from model.cnn_exp import Net, testset
from mrfi.injector import ModuleInjector

import yaml
import logging
import torch
import numpy as np

#configurator = Configurator(Net(), ['conv'])
#configurator.save('cnn_exp.yaml')

def default_inject():
    with open("configs/cnn_exp.yaml") as f:
        config = yaml.full_load(f)

    net=Net()
    net.load_state_dict(torch.load('_data/cifar_vgg.pth'))
    net.eval()

    logging.basicConfig(level=logging.NOTSET)
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

    logging.basicConfig(level=logging.NOTSET)
    FI_network = ModuleInjector(net, config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    data=iter(testloader)

    total = 1000

    for i in range(total):
        images, labels = next(data)
        FI_network(images, golden=True)
        FI_network(images)

    observes=FI_network.get_observes()
    for name, value in observes.items():
        print(name, np.sqrt(value/total))

quan_inject()

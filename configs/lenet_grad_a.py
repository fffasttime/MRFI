from email.mime import image
import torch
import torch.nn as nn
import numpy as np
from torch import autograd

from model.lenet_cifar import Net, testset

act = None

def observer_hook_func(input):
    global act
    act=input

def experiment(total = 10000):
    torch.set_num_threads(8)

    net=Net(True)
    device = torch.device("cpu") # lenet uses cpu only
    net.to(device)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    data=iter(testloader)
    gradsum = np.zeros([3072,3])
    
    net.conv1.register_forward_pre_hook(lambda mod, input: observer_hook_func(input[0]))

    for i in range(total):
        images, labels = next(data)
        images.requires_grad_()
        images=images.to(device)
        net.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)

        grad = autograd.grad(loss, act, create_graph=True)[0]
        grad2 = autograd.grad(grad, act, torch.ones_like(grad))[0]
        gradsum[:,0]+=act.detach().abs().numpy().reshape(-1)/total
        gradsum[:,1]+=grad.detach().abs().numpy().reshape(-1)/total
        gradsum[:,2]+=grad2.abs().numpy().reshape(-1)/total
    
    np.savetxt('grad2.txt', gradsum, fmt="%.5f")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd
import matplotlib.pyplot as plt
from mrfi.flip_mode import flip_int_highest

from model.lenet_cifar import Net, testset


def layerwise_quantization(x, bit_width, dynamic_range):
    down_limit, up_limit = -1<<bit_width-1, (1<<bit_width-1)-1
    x += dynamic_range
    x *= (up_limit - down_limit) / (dynamic_range*2)
    x += down_limit
    x.clamp_(down_limit, up_limit)
    x.round_()

def layerwise_dequantization(x, bit_width, dynamic_range):
    down_limit, up_limit = -1<<bit_width-1, (1<<bit_width-1)-1
    x -= down_limit
    x *= (dynamic_range*2)/(up_limit - down_limit)
    x -= dynamic_range

def experiment(total = 10000):
    torch.set_num_threads(8)

    net=Net(True)
    device = torch.device("cpu") # lenet uses cpu only
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    data=iter(testloader)
    gradsum = np.zeros([2400,3])
    gradsum[:,0] = net.conv2.weight.detach().abs().numpy().reshape(-1)

    flipw=net.conv2.weight.detach().clone()
    layerwise_quantization(flipw, 8, 8)
    for p in range(2400):
        flipw.view(-1)[p] = flip_int_highest(flipw.view(-1)[p].item(), 8)
    layerwise_dequantization(flipw, 8, 8)
    sign = torch.sign(flipw - net.conv2.weight.detach())
    
    gradhistory = np.zeros(total)

    for i in range(total):
        images, labels = next(data)
        images=images.to(device)
        net.zero_grad()
        outputs = net(images)
        #outputs = F.softmax(outputs)
        #labels = nn.functional.one_hot(labels.to(torch.long), num_classes = 10).float()
        loss = criterion(outputs, labels)
        grad = autograd.grad(loss, net.conv2.weight, create_graph=True)[0]
        grad2 = autograd.grad(grad, net.conv2.weight, torch.ones_like(grad))[0]
        #gradsum[:,1]+=grad.detach().abs().numpy().reshape(-1)/total
        gradsum[:,1]+=((F.relu(sign))*grad.detach().abs()).numpy().reshape(-1)/total
        gradhistory[i] = grad.detach().view(-1)[3].item()
        gradsum[:,2]+=grad2.numpy().reshape(-1)/total
    
    #plt.hist(gradhistory, 30, (-0.5, 0.5))
    #plt.show()

    np.savetxt('grad3.txt', gradsum, fmt="%.5f")

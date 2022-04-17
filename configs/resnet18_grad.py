from email.mime import image
from torch import autograd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mrfi.flip_mode import flip_int_highest
from model.resnet18 import Net, testset

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    layerw = net.layer1[1].conv2.weight

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    data=iter(testloader)
    gradsum = np.zeros([36864,6])
    gradsum[:,0] = layerw.cpu().detach().abs().numpy().reshape(-1)
    print('wmax =', np.max(gradsum[:, 0]), 'wstd =', np.std(gradsum[:, 0]))
    
    flipw=layerw.detach().clone()
    layerwise_quantization(flipw, 16, 4)
    for p in range(36864):
        flipw.view(-1)[p] = flip_int_highest(flipw.view(-1)[p].item(), 8)
    layerwise_dequantization(flipw, 16, 4)
    sign = torch.sign(flipw - layerw.detach()).cpu()
    gradsum[:,5] = sign.numpy().reshape(-1)
    
    gradhistory = np.zeros(total)

    loss_data = []
    acc_data = []

    net.eval()

    #acc=0
    for i in range(total):
        images, labels = next(data)
        images=images.to(device)
        labels=labels.to(device)
        net.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        #loss_data.append(loss)
        #print(np.argmax(outputs.cpu().detach().numpy()[0]), labels.cpu().numpy()[0])
        #acc+=(np.argmax(outputs.cpu().detach().numpy()[0])==labels[0].cpu().numpy())

        #acc_data.append(np.argmax(outputs.cpu().detach().numpy()[0])==labels.cpu().numpy()[0])
        #print(criterion(outputs[:1], labels[:1]), outputs[:1], labels[:1])
        grad = autograd.grad(loss, layerw, create_graph=True)[0]
        grad2 = autograd.grad(grad, layerw, torch.ones_like(grad))[0]
        gradsum[:,1]+=grad.cpu().detach().abs().numpy().reshape(-1)/total
        gradsum[:,2]+=grad2.cpu().numpy().reshape(-1)/total
        gradsum[:,3]+=((F.relu(sign))*grad.cpu().detach().abs()).numpy().reshape(-1)/total
        gradsum[:,4]+=((F.relu(sign))*grad2.cpu()).numpy().reshape(-1)/total
        if (i%100==0):
             print(i, end=' ', flush=True)
    
    #print("%.2f%%"%(acc/total*100))
    #plt.hist(gradhistory, 30, (-0.5, 0.5))
    #plt.show()
    #print(acc_data, loss_data)

    np.savetxt('_logs/resnet18_grad6_l1c4.txt', gradsum, fmt="%.5f")

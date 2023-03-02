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

def experiment(total = 50000):
    torch.set_num_threads(8)

    net=Net(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    layers = [
    net.conv1.weight,
    net.layer1[0].conv1.weight, 
    net.layer1[0].conv2.weight, 
    net.layer1[1].conv1.weight, 
    net.layer1[1].conv2.weight,
    net.layer2[0].conv1.weight, 
    net.layer2[0].conv2.weight, 
    net.layer2[1].conv1.weight, 
    net.layer2[1].conv2.weight,
    net.layer3[0].conv1.weight, 
    net.layer3[0].conv2.weight, 
    net.layer3[1].conv1.weight, 
    net.layer3[1].conv2.weight,
    net.layer4[0].conv1.weight, 
    net.layer4[0].conv2.weight, 
    net.layer4[1].conv1.weight, 
    net.layer4[1].conv2.weight,
    ]

    for layerw in layers:
        print(torch.max(torch.abs(layerw)).item())

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    data=iter(testloader)

    loss_data = []
    acc_data = []

    net.eval()

    gradsum=np.zeros(len(layers))
    gradall=[np.zeros(w.shape) for w in layers]
    
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
        grad = autograd.grad(loss, layers)
        
        for l in range(len(layers)):
            gradsum[l]+=torch.sum(grad[l].cpu().detach().abs()).numpy().reshape(-1)/total
            gradall[l]+=grad[l].cpu().detach().abs().numpy()/total
            
        if (i%500==0):
            print(i, end=' ', flush=True)
    
    for d in gradsum:
        print(d)
    #print("%.2f%%"%(acc/total*100))
    #plt.hist(gradhistory, 30, (-0.5, 0.5))
    #plt.show()
    #print(acc_data, loss_data)

    #np.savetxt('_logs/resnet18_grad6_l1c4.txt', gradsum, fmt="%.5f")
    np.savez_compressed('resnet18_gradallweight.npz', *gradall)

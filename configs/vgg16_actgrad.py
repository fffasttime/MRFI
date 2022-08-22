from email.mime import image
from torch import autograd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mrfi.flip_mode import flip_int_highest
from model.vgg16 import Net, testset

grad = None

def observer_hook_func(output):
    global grad
    grad=output

def experiment(total = 10000):
    torch.set_num_threads(8)

    for convlayer in range(13):
        net=Net(True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        layers = list(net.features)
        layers = list(filter(lambda x:str(x)[:4]=='Conv', layers))
        assert(len(layers)==13) # 13 conv layers

        layer = layers[convlayer] # get conv 0

        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        data=iter(testloader)

        net.eval()

        gradsum = None
        layer.register_backward_hook(lambda mod, input, output: observer_hook_func(output[0]))
        pbar = tqdm(range(total))
        for i in pbar:
            images, labels = next(data)
            images.requires_grad_()
            images=images.to(device)
            labels=labels.to(device)
            net.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            loss.backward()

            global grad

            if gradsum is None:
                gradsum = grad.detach().abs().cpu().numpy().reshape(-1)/total
            else:
                gradsum += grad.detach().abs().cpu().numpy().reshape(-1)/total

        print(grad.shape)
        np.savetxt('_logs/vgg16_actgrad_conv%d.txt'%(convlayer+1), gradsum)

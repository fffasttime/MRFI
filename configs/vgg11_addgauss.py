import math
import yaml
import logging
import torch
import numpy as np

import mrfi.observer
from model.vgg11 import Net, testset
from mrfi.injector import ModuleInjector

gauss_std=[0.142,0.1639,0.1854,0.2083,0.2327,0.2683,0.3066,0.3475,0.399,0.4537,0.5195,0.5927,0.687,0.7869,0.9055,1.0454,1.1957,1.3652,1.5685,1.7813,2.0228,2.2863,2.5848,2.9121,3.2927,3.7297,4.2213,4.7989,5.4389,6.1453,6.8609]
gauss_std=np.array(gauss_std)*math.sqrt(2)

def experiment(total = 10000):
    torch.set_num_threads(16)
    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    acc=np.zeros(len(gauss_std))

    data=iter(testloader)
    for _ in range(total):
        images, labels = next(data)
        images = images.to(device)
        out=net(images).cpu().detach().numpy()
        for i, std in enumerate(gauss_std):
            out_p = out + np.random.randn(1000)*std
            acc[i]+=(np.argmax(out_p[0])==labels.numpy()[0])

    for v in acc:
        print('%.2f%%'%(v/total*100))
import yaml
import logging
import torch
import numpy as np

from model.vgg11 import Net, testset

def exp(total = 10000):
    torch.set_num_threads(16)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    data=iter(testloader)
    np.set_printoptions(suppress=True, precision = 2)
    
    acc=0
    res = []
    for i in range(total):
        images, labels = next(data)
        images=images.to(device)
        out=net(images).detach().cpu().numpy()[0]
        l = labels.numpy()[0]
        margin = np.sort(np.concatenate((out[0:l], out[l+1:1000]))-out[l])
        res.append(margin)
        acc+=(np.argmax(out)==l)

    print("%.2f%%"%(acc/total*100), flush=True)
    np.save('margin.npy', np.array(res))

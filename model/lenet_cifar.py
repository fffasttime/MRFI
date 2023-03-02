import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


try:
    import os
    datapath = os.environ['DATASETS'] + '/cifar10'
except Exception:
    datapath = './_data'

trainset = torchvision.datasets.CIFAR10(root=datapath, train=True,
                                        download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=datapath, train=False,
                                       download=False, transform=transform)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Net(nn.Module):
    def __init__(self, trained = False):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        if trained:
            self.load_state_dict(torch.load('./_data/cifar10.pth'))

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    net = Net()
    net.to(device)
    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    net.train()

    for epoch in range(10):
    
        running_loss = 0.
        batch_size = 100
        
        for i, data in enumerate(
                torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True), 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))
    
    print('Finished Training')

    torch.save(net.state_dict(), 'cifar10.pth')

def test():
    net=Net()
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False)
    net.load_state_dict(torch.load('cifar10.pth'))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
trainset = torchvision.datasets.CIFAR10(root='./_data', train=True,
                                        download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./_data', train=False,
                                       download=False, transform=transform)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

def conv(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)

import numpy as np

golden=[]
cur=0
layer_diff, layer_sum = None, None

def sse(x):
    return np.sum(x*x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = conv(3, 64, 3, padding = 1)
        self.conv2 = conv(64, 64, 3, padding = 1)
        self.conv3 = conv(64, 128, 3, padding = 1)
        self.conv4 = conv(128, 128, 3, padding = 1)
        self.conv5 = conv(128, 256, 3, padding = 1)
        self.conv6 = conv(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn11 = nn.BatchNorm2d(64)
        self.bn21 = nn.BatchNorm2d(128)
        self.bn31 = nn.BatchNorm2d(256)
        self.bn12 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(128)
        self.bn32 = nn.BatchNorm2d(256)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)
        
    def forward(self,x):
        x = self.bn11(F.relu(self.conv1(x)))
        x = self.bn12(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.bn21(F.relu(self.conv3(x)))
        x = self.bn22(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.bn31(F.relu(self.conv5(x)))
        x = self.bn32(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout10(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import torch.optim as optim
    net = Net()
    net.to(device)
    
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

    torch.save(net.state_dict(), 'cifar_vgg.pth')

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False)
    net.load_state_dict(torch.load('cifar_vgg.pth'))
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
    
    print('Accuracy of the network on the 10000 test images: %.1f %%' % (
        100 * correct / total))

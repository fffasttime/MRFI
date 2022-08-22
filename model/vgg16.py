import torch
from torch.nn.functional import normalize
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

Net=models.vgg16

normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

tf=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

testset=datasets.ImageFolder('~/dataset/val', tf)

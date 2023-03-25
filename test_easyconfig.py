import torch
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import mrfi
from mrfi import MRFI, EasyConfig

fi_model = MRFI(models.resnet18(pretrained = True), 
                EasyConfig.load_file('configs/default_fi.yaml'))

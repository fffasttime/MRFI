import torch
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import mrfi
from mrfi import MRFI, EasyConfig

tf=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

fi_model = MRFI(models.resnet50(pretrained = True), 
                EasyConfig.load_file('easyconfigs/default_fi.yaml'))

fi_model.save_config('detailconfigs/resnet50_default_fi.yaml')
testset=datasets.ImageFolder('~/dataset/val', tf)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    fi_model.cuda()
else:
    device = torch.device('cpu')

# method 1
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

acc_golden, acc_fi = 0, 0

for images, labels in testloader:
    images=images.to(device)
    
    # fault free run
    with fi_model.golden():
        outs_golden=fi_model(images)
    # fault inject run
    outs=fi_model(images)
    acc_golden+=(outs_golden.argmax(1)==labels).sum().item()
    acc_fi+=(outs.argmax(1)==labels).sum().item()

print(f'{len(acc_golden)} images, acc_golden {acc_golden}, acc_fi {acc_fi}')

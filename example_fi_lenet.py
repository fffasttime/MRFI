import torch
from model.lenet_cifar import testset, Net
from mrfi import MRFI, EasyConfig
import sys

configfilename = 'default_fi'
if len(sys.argv)>1:
    configfilename = sys.argv[1]

fi_model = MRFI(Net(trained=True), EasyConfig.load_file('easyconfigs/%s.yaml'%configfilename))
fi_model.save_config('detailconfigs/lenet_%s.yaml'%configfilename)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    fi_model.cuda()
else:
    device = torch.device('cpu')

# method 1
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

acc_golden, acc_fi = 0, 0

cfg = fi_model.get_configs('', 'weights.0.selector', False)
print(cfg)

for i in range(2):
    
    fi_model.observers_reset()
    for images, labels in testloader:
        images=images.to(device)
        
        # fault free run
        with fi_model.golden_run():
            outs_golden=fi_model(images)
        # fault inject run
        outs_fi=fi_model(images)

        acc_golden+=(outs_golden.argmax(1)==labels).sum().item()
        acc_fi+=(outs_fi.argmax(1)==labels).sum().item()

    print(f'{len(testset)} images, acc_golden {acc_golden}, acc_fi {acc_fi}')
    print(fi_model.observers_result())

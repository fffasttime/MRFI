
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density
import matplotlib.pyplot as plt
from torchvision.models import resnet18

econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
fi_model = MRFI(resnet18(pretrained = True), econfig)

selector_cfg = fi_model.get_configs('', 'activation.0.selector', False)

BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(1000, batch_size = 128), logspace_density(-6, -3, 3))

print(Acc)

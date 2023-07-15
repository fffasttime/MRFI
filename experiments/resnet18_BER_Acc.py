
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density, Acc_golden, benchmark_range
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch

econfig = EasyConfig.load_file('easyconfigs/fxp_fi.yaml')
fi_model = MRFI(resnet18(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
quantization_cfg = fi_model.get_activation_configs('quantization.args')
errormode_cfg = fi_model.get_activation_configs('error_mode.args')

batch_size = 128
n_images = 1000

quantization_cfg.integer_bit = 2
quantization_cfg.decimal_bit = 12
errormode_cfg.bit_width = 14

Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size))
BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size))

print(Acc)

quantization_cfg.integer_bit = 3
quantization_cfg.decimal_bit = 13
errormode_cfg.bit_width = 16

Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size))
BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size))

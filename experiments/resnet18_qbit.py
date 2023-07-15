
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density, Acc_golden
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch

econfig = EasyConfig.load_file('easyconfigs/fxp_fi.yaml')
econfig.set_error_mode(0, {'method':'IntFixedBitFlip', 'bit':16, 'bit_width': 17})
econfig.set_quantization(0, {'integer_bit':4, 'decimal_bit':12}, True)
fi_model = MRFI(resnet18(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
errormode_cfg = fi_model.get_activation_configs('error_mode.args')

batch_size = 128
n_images = 10000

# print('Golden acc', Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size)))

result = {}

for bit in range(16, 9, -1):
    errormode_cfg.bit = bit
    BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), logspace_density(-7, -2))
    result['Q_b%d'%bit] = Acc
    print(bit, BER, Acc)

import numpy as np
np.savez('result/resnet18_fixbit.npz', BER = BER, **result)

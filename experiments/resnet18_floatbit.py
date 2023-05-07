
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density, Acc_golden, benchmark_range
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'FloatFixBitFlip'
econfig.faultinject[0]['error_mode']['floattype'] = 'float16'
econfig.faultinject[0]['error_mode']['bit'] = 31
fi_model = MRFI(resnet18(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_configs('activation.0.selector')
errormode_cfg = fi_model.get_configs('activation.0.error_mode.args')

batch_size = 128
n_images = 10000

print('Golden acc', Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size)))

result = {}

for bit in range(15, -1, -1):
    errormode_cfg.bit = bit
    BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), logspace_density(-9, -3))
    result['f16_b%d'%bit] = Acc
    print(bit, BER, Acc)

import numpy as np
np.savez('result/resnet18_float16bit.npz', BER = BER, **result)

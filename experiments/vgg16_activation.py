
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density, Acc_golden, get_activation_info
from torchvision.models import vgg16

batch_size = 128
n_images = 10000

range = logspace_density(-9, -3)

econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
econfig.faultinject[0]['quantization']['scale_factor'] = 2
fi_model = MRFI(vgg16(pretrained = True).cuda().eval(), econfig)
ranges = get_activation_info(fi_model, make_testloader(128, batch_size=128), 'MaxAbs', pre_hook = True, module_type = ['Conv', 'Linear'])
selector_cfg = fi_model.get_activation_configs('selector')
q_args = fi_model.get_activation_configs('quantization.args')
q_args.dynamic_range = list(ranges.values())

BER, Qlw = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), range)

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
fi_model = MRFI(vgg16(pretrained = True).cuda().eval(), econfig)


print('golden_Acc', Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size)))
selector_cfg = fi_model.get_activation_configs('selector')
BER, fl32 = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), range, bit_width=32)


econfig = EasyConfig.load_file('easyconfigs/fxp_fi.yaml')
del econfig.faultinject[0]['module_name']
econfig.faultinject[0]['module_type'] = ['Conv2d', 'Linear']
fi_model = MRFI(vgg16(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
quantization_cfg = fi_model.get_activation_configs('quantization.args')
errormode_cfg = fi_model.get_activation_configs('error_mode.args')

quantization_cfg.integer_bit = 6
quantization_cfg.decimal_bit = 10
errormode_cfg.bit_width = 17

BER, fxp2 = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), range, bit_width=17)

quantization_cfg.integer_bit = 7
quantization_cfg.decimal_bit = 9
errormode_cfg.bit_width = 17

BER, fxp3 = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), range, bit_width=17)

import numpy as np

np.savez('result/vgg16_activation.npz', BER = BER, fxp2 = fxp2, fxp3 = fxp3, fl32 = fl32, Qlw = Qlw)

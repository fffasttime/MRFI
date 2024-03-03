
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from torchvision.models import vgg11
import torch

econfig = EasyConfig.load_preset('float_fi')
econfig.faultinject[0]['error_mode']['method'] = 'FloatFixedBitFlip'
econfig.faultinject[0]['error_mode']['floattype'] = 'float32'
econfig.faultinject[0]['error_mode']['bit'] = 31
econfig.faultinject[0]['error_mode']['suppress_invalid'] = True
econfig.set_module_used(0, module_fullname=['features.14', 'features.15'])
econfig.set_selector(0, {'method':'RandomPositionByNumber', 'n':1, 'per_instance':True})
# econfig.faultinject[0]['type'] = 'weight'
# econfig.faultinject[0]['type'] = 'activation_out'
fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
errormode_cfg = fi_model.get_activation_configs('error_mode.args')

batch_size = 128
n_images = 10000

result = []
device = next(fi_model.parameters()).device

bits = list(range(24, 32))
bits.reverse()

for bit in bits:
    test_loader = make_testloader(n_images, batch_size = batch_size)
    errormode_cfg.bit = bit

    sum_drop = 0
    
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        with fi_model.golden_run():
            out_g = fi_model(imgs).cpu()
            acc_g = out_g.argmax(1)==labels
        
        out = fi_model(imgs).cpu()
        out[torch.isnan(out)] = 0
        acc = out.argmax(1)==labels

        sum_drop += torch.sum((acc_g == 1) & (acc == 0))
    
    print(bit, sum_drop.item())
    result.append(sum_drop.item())

import numpy as np

with open('result/vgg16_float32_bit_errcnt.txt', 'w') as f:
    print('bits', bits, file = f)
    print('errs', result, file = f)

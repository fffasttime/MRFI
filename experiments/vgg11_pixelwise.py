
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import get_activation_info, observeFI_experiment_plus, Acc_experiment
from torchvision.models import vgg11
import torch
import numpy as np

batch_size = 100
n_images = 100

config = """
faultinject:
  - type: activation_out
    quantization:
      method: SymmericQuantization
      dynamic_range: auto
      bit_width: 16
    enabled: True
    selector:
      method: FixedPixelByNumber
      pixel: (0,0)
      n: 10
      per_instance: True
    error_mode:
      method: IntSignBitFlip
      bit_width: 16

    module_name: features
"""

fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), EasyConfig.load_string(config))
selector_cfg = fi_model.get_activation_configs('selector', out=True)
activation_cfg = fi_model.get_activation_configs(is_out=True)

activation_cfg.enabled = False

shapes = get_activation_info(fi_model, make_testloader(1), method='Shape', module_type = 'Conv')
print(shapes)
shapes = list(shapes.values())

results = {}

for layer in [3, 5]:
    activation_cfg.enabled = False
    activation_cfg[layer].enabled = True
    print(shapes[layer])
    pixels = shapes[layer][2], shapes[layer][3]
    rmses = np.empty(pixels)

    for i in range(pixels[0]):
        for j in range(pixels[1]):
          selector_cfg[layer].args.pixel = (i, j)
          rmse = observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), module_fullname = '')
          rmse = list(rmse.values())[0]
          print(layer, i, j, rmse)
          rmses[i, j] = rmse

    results['conv%d'%layer] = rmses

np.savez('result/vgg11_pixelwise_o.npz', **results)

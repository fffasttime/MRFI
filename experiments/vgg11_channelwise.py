
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import observeFI_experiment_plus, get_activation_info
from torchvision.models import vgg11
import numpy as np

batch_size = 100
n_images = 1000

econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.set_module_used(0, 'Conv')
econfig.set_selector(0, {'method':'SelectedDimRandomPositionByNumber', 'n':batch_size * 10, 'channel': [0]})

fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), econfig)
selector_cfg = fi_model.get_activation_configs('selector')
action_cfg = fi_model.get_activation_configs()

action_cfg.enabled = False

shapes = get_activation_info(fi_model, make_testloader(1), method='Shape', module_type = 'Conv', pre_hook = True)
print(shapes)
shapes = list(shapes.values())

results = {}

for layer in range(1,4):
    action_cfg.enabled = False
    action_cfg[layer].enabled = True
    channels = shapes[layer][1]
    rmses = np.empty((channels,))

    for i in range(channels):
        selector_cfg[layer].args.channel = [i]
        rmse = observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), module_fullname = '')
        rmse = list(rmse.values())[0]
        print(layer, i, rmse)
        rmses[i] = rmse

    results['conv%d'%layer] = rmses

np.savez('result/vgg11_channelwise.npz', **results)

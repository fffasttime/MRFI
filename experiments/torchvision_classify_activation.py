
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import Acc_golden, Acc_experiment, get_activation_info
from torchvision.models import alexnet
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.vision_transformer import vit_b_16

batch_size = 10000
n_images = 128

# ViT
econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
econfig.faultinject[0]['quantization']['scale_factor'] = 1
econfig.faultinject[0]['selector']['rate'] = 1.6e-4
econfig.set_module_used(0, module_name = ['mlp'], module_type = ['Conv'])

fi_model = MRFI(vit_b_16(pretrained = False).eval(), econfig)
print(fi_model)

ranges = get_activation_info(fi_model, make_testloader(128, batch_size=128), 'MaxAbs', pre_hook = True, module_name = ['mlp'], module_type = ['Conv'])
selector_cfg = fi_model.get_activation_configs('selector')
q_args = fi_model.get_activation_configs('quantization.args')
q_args.dynamic_range = list(ranges.values())
Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size))
Acc_experiment(fi_model, make_testloader(n_images, batch_size = batch_size))

# AlexNet
econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
econfig.faultinject[0]['quantization']['scale_factor'] = 1
econfig.faultinject[0]['selector']['rate'] = 1.6e-4

fi_model = MRFI(alexnet(pretrained = True).cuda().eval(), econfig)

ranges = get_activation_info(fi_model, make_testloader(128, batch_size=128), 'MaxAbs', pre_hook = True, module_type = ['Conv', 'Linear'])
selector_cfg = fi_model.get_activation_configs('selector')
q_args = fi_model.get_activation_configs('quantization.args')
q_args.dynamic_range = list(ranges.values())
Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size))
Acc_experiment(fi_model, make_testloader(n_images, batch_size = batch_size))

# MobileNet
econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
econfig.faultinject[0]['quantization']['scale_factor'] = 1
econfig.faultinject[0]['selector']['rate'] = 1.6e-4
econfig.set_module_used(0, module_type = ['Conv', 'Linear'])

fi_model = MRFI(mobilenet_v2(pretrained = True).cuda().eval(), econfig)

ranges = get_activation_info(fi_model, make_testloader(128, batch_size=128), 'MaxAbs', pre_hook = True, module_type = ['Conv', 'Linear'])
selector_cfg = fi_model.get_activation_configs('selector')
q_args = fi_model.get_activation_configs('quantization.args')
q_args.dynamic_range = list(ranges.values())
Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size))
Acc_experiment(fi_model, make_testloader(n_images, batch_size = batch_size))


from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import Acc_golden, Acc_experiment
from torchvision.models import vgg16

batch_size = 128
n_images = 10000

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
econfig.set_selector(0, {'method':'RandomPositionByNumber', 'n':1, 'per_instance': True})
econfig.set_module_used(0, module_fullname=['features.9','features.10','features.11'])
fi_model = MRFI(vgg16(pretrained = True).cuda().eval(), econfig)

print('golden_Acc', Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size)))
selector_cfg = fi_model.get_activation_configs('selector')

nums = [2,3,4,5,6,7,10,15,20,30,40,50,60]

for num in nums:
    selector_cfg.args.n = num
    result = Acc_experiment(fi_model, make_testloader(n_images, batch_size = batch_size))
    print('%d\t%.2f'%(num, result*100))

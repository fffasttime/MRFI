'''Compare speed of poisson_sample'''
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density, Acc_golden
from torchvision.models import resnet18
import torch
import time
from torch.utils.data import Dataset, DataLoader

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
econfig.faultinject[0]['error_mode']={'method':'FloatFixBitFlip', 'floattype':'float32', 'bit':30}
fi_model = MRFI(resnet18(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
errormode_cfg = fi_model.get_activation_configs('error_mode.args')

batch_size = 128
n_images = 10000

class FakeDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    def __len__(self):
        return n_images
    def __getitem__(self, index):
        return torch.zeros([3,224,224], device = 'cuda'), 0

fake_testloader = DataLoader(FakeDataset(), batch_size=batch_size)

print('Warming up', Acc_golden(fi_model, fake_testloader))

'''
start_time = time.time()
print('Golden acc', Acc_golden(fi_model, make_testloader(n_images, batch_size = batch_size)))
print('Time: ', time.time() - start_time) # 91.204

start_time = time.time()
BER, Acc_True = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), [1e-7])
print('Time: ', time.time() - start_time) # 91.290
'''

start_time = time.time()
print('Golden acc', Acc_golden(fi_model, fake_testloader))
print('Time: ', time.time() - start_time)

start_time = time.time()
BER, Acc_True = BER_Acc_experiment(fi_model, selector_cfg, fake_testloader, [1e-7])
print('Time: ', time.time() - start_time)


'''
selector_cfg.args.poisson_sample = True
BER, Acc_True = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), logspace_density(-10, -7, 5))
print('pos true', BER, Acc_True)


selector_cfg.args.poisson_sample = False
BER, Acc_False = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), logspace_density(-10, -7, 5))
print('pos false', BER, Acc_False)

import numpy as np
np.savez('result/poisson_sample.npz', BER = BER, Acc_True = Acc_True, Acc_False = Acc_False)
'''

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
econfig.faultinject[0]['error_mode']={'method':'FloatFixBitFlip', 'floattype':'float32', 'bit':30}
econfig.faultinject[0]['selector'] = {'method': 'RandomPositionByRate_classic', 'rate':1e-7}

fi_model = MRFI(resnet18(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_activation_configs('selector')

start_time = time.time()
BER, Acc_GT = BER_Acc_experiment(fi_model, selector_cfg, fake_testloader, [1e-7])
print('Time: ', time.time() - start_time)

BER, Acc_GT = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), logspace_density(-10, -7, 5))
print(Acc_GT)
torch.savez('temp.npz', Acc_GT = Acc_GT)

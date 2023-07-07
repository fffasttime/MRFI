
from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import Acc_experiment, Acc_golden
from torchvision.models import vgg11
import time
from torch.utils.data import Dataset, DataLoader
import torch

batch_size = 128
n_images = 50000


torch.set_num_threads(4)

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
econfig.set_selector(0, {'method':'RandomPositionByNumber', 'n':1, 'per_instance':True})

fi_model = MRFI(vgg11(pretrained = True).eval(), econfig)
clean_model = vgg11(pretrained = True).eval()
#fi_model.cuda()
#clean_model.cuda()

#print(Acc_experiment(clean_model, make_testloader(n_images, batch_size = batch_size)))
#print(Acc_experiment(fi_model, make_testloader(n_images, batch_size = batch_size)))

class FakeDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    def __len__(self):
        return n_images
    def __getitem__(self, index):
        return torch.zeros([3,224,224], device = 'cuda'), 0

fake_testloader = DataLoader(FakeDataset(), batch_size=batch_size)

def timeit(func, repeat = 3):
    time_sum = 0.0
    for i in range(repeat):
        t0, t0f = time.perf_counter(), time.time()
        func()
        print(i, time.perf_counter()-t0, time.time()-t0f)
        time_sum += time.perf_counter()-t0
    print('ave ', time_sum / repeat,'s')

timeit(lambda: Acc_experiment(clean_model, fake_testloader))
#timeit(lambda: Acc_experiment(fi_model, fake_testloader))
timeit(lambda: Acc_experiment(fi_model, fake_testloader))

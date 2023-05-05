from dataset.lenet_cifar import make_testloader, Net
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density, benchmark_range
import matplotlib.pyplot as plt

econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'

fi_model = MRFI(Net(trained=True), econfig)

selector_cfg = fi_model.get_configs('', 'activation.0.selector', False)


res = benchmark_range(fi_model, make_testloader(1000, batch_size = 128), ['conv'])

print(res)

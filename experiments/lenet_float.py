from dataset.lenet_cifar import make_testloader, Net
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density
import matplotlib.pyplot as plt

econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
fi_model = MRFI(Net(trained=True), econfig)

selector_cfg = fi_model.get_configs('', 'activation.0.selector', False)

BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(1000, batch_size = 128), logspace_density(-6, -3, 3))

print(Acc)
plt.plot(BER, Acc)

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
fi_model = MRFI(Net(trained=True), econfig)

selector_cfg = fi_model.get_configs('', 'activation.0.selector', False)
error_mode_cfg = fi_model.get_configs('', 'activation.0.error_mode.args', False)

error_mode_cfg.floattype = 'float16'
BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(1000, batch_size = 128), logspace_density(-6, -3, 3))

print(Acc)
plt.plot(BER, Acc)

error_mode_cfg.floattype = 'float32'
BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(1000, batch_size = 128), logspace_density(-6, -3, 3))

print(Acc)
plt.plot(BER, Acc)

error_mode_cfg.floattype = 'float64'
BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(1000, batch_size = 128), logspace_density(-6, -3, 3))

print(Acc)
plt.plot(BER, Acc)

plt.xscale('log')
plt.xlabel('BER')
plt.ylabel('Acc')

legend = ['fixed', 'float16', 'float32', 'float64']
plt.legend(legend)

plt.show()

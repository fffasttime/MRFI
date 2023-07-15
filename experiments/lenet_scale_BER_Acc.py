from dataset.lenet_cifar import make_testloader, Net
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment, logspace_density
import matplotlib.pyplot as plt

econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
fi_model = MRFI(Net(trained=True), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
quantization_args = fi_model.get_activation_configs('quantization.args')

print(selector_cfg, quantization_args)
legend = []

for scale_factor in [0.35, 0.5, 0.75, 1, 2, 4]:  # quantization scale
    quantization_args.scale_factor = scale_factor  # auto boardcast to all scale_factors
    BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(1000, batch_size = 128), logspace_density(-6, -3, 3))

    print(Acc)
    plt.plot(BER, Acc)
    plt.xscale('log')
    legend.append('scale_factor = %.2f'%scale_factor)

plt.legend(legend)
plt.xlabel('BER')
plt.ylabel('Acc')
plt.ylim(0.2, 0.7)
plt.show()

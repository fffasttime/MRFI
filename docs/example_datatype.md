# Datatype/bit experiemt

## Datatype experiment

The following code compares the fault tolerance difference between quantized integer type and 3 float types,
and visualze them using matplotlib.

```python
econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
fi_model = MRFI(Net(trained=True), econfig)

selector_cfg = fi_model.get_activation_configs('selector')

BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(1000, batch_size = 128), logspace_density(-6, -3, 3))

print(Acc)
plt.plot(BER, Acc)

econfig = EasyConfig.load_file('easyconfigs/float_fi.yaml')
fi_model = MRFI(Net(trained=True), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
error_mode_cfg = fi_model.get_activation_configs('error_mode.args')

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
```

## Bit experiment on quantized value

The following code conduct experiment on high bits of fix point quantized model, 
in order to study bit sensitivity.

To distinguish bit sensitivity of float point model, just remove the quantization and use a float error mode method.

```python
econfig = EasyConfig.load_file('easyconfigs/fxp_fi.yaml')
econfig.set_error_mode(0, {'method':'IntFixedBitFlip', 'bit':16, 'bit_width': 17})
econfig.set_quantization(0, {'integer_bit':4, 'decimal_bit':12}, True)
fi_model = MRFI(resnet18(pretrained = True).cuda().eval(), econfig)

selector_cfg = fi_model.get_activation_configs('selector')
errormode_cfg = fi_model.get_activation_configs('error_mode.args')

result = {}

for bit in range(16, 9, -1):
    errormode_cfg.bit = bit
    BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, make_testloader(n_images, batch_size = batch_size), logspace_density(-7, -2))
    result['Q_b%d'%bit] = Acc
    print(bit, BER, Acc)

import numpy as np
np.savez('result/resnet18_fixbit.npz', BER = BER, **result)
```

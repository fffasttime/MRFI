# Basic Usage

## Examples

### Observe network weights and activations

We provide `get_activation_info` and `get_weight_info` in `mrfi.experiment`.
They can be used to observe the shape, distribution, and sampling & visualization of network data.

[>> Example of basic observe on LeNet](basic_observe.md)

### A coarse-grained configuation fault inject experiment
For example, the following code perform a quantized random integer bit flip injection on LeNet, 
and find the relation between bit error rate (BER) and classification accuracy.

```python title="LeNet default fault injection"
from dataset.lenet_cifar import make_testloader, LeNet
from mrfi import MRFI, EasyConfig
from mrfi.experiment import BER_Acc_experiment

testloader = make_testloader(1000, batch_size = 128) # test on 1000 cifar-10 images

# Create fault inject model
fi_model = MRFI(network = LeNet(trained=True).eval(), 
                EasyConfig.load_file('easyconfigs/default_fi.yaml'))

################### A Simple acccuracy experiment #####################
# Test accuracy under fault injection with select rate = 1e-3 which specified in "default_fi.yaml"
print('FI Acc: ', Acc_experiment(fi_model, dataloader))
# Test accuracy w/o fault inject
with fi_model.golden_run():
    print('golden run Acc: ', Acc_experiment(fi_model, dataloader))

######################## BER_Acc_experiment ###########################
# Get selector handler because BER_Acc_experiment needs to modify selection rate in experiment
selector_cfg = fi_model.get_configs('activation.0.selector')
BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, testloader, [1e-6, 1e-5, 1e-4, 1e-3])
print('Bit error rate and accuracy: ', BER, Acc)
```

# Advanced fine-grained config

## Modify ConfigTree directly

Content of detail config tree can be visited and modified by `FI_config` attribute on a module(layer).

The following code in python terminal demonstrates how to visit and modify one property in ConfigTree.

```python title="visit and modify property in ConfigTree node"
>>> from mrfi import MRFI, EasyConfig
>>> from torchvision.models import resnet18
>>> from pprint import pprint
>>> econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
>>> fi_model = MRFI(resnet18(pretrained = True).eval(), econfig)
>>>
>>> print(fi_model.conv1.FI_config)
<'MODULE' node 'model.sub_modules.conv1'>
>>>
>>> # We use `state_dict()` to visualize a detail config tree node
>>> pprint(fi_model.conv1.FI_config.state_dict())
{'activation': {0: {'enabled': True,
                    'error_mode': {'args': {'bit_width': 16},
                                   'method': 'IntSignBitFlip'},
                    'quantization': {'args': {'bit_width': 16,
                                              'dynamic_range': 'auto'},
                                     'layerwise': True,
                                     'method': 'SymmericQuantization'},
                    'selector': {'args': {'poisson': True, 'rate': '1e-3'},
                                 'method': 'RandomPositionByRate'}}},
 'activation_out': {},
 'observers': {},
 'observers_pre': {},
 'sub_modules': {},
 'weights': {}}
>>>
>>> # Show and modify "enabled" property
>>> print(fi_model.conv1.FI_config.activation[0].enabled)
True
>>> fi_model.conv1.FI_config.activation[0].enabled = False
>>> print(fi_model.conv1.FI_config.activation[0].enabled)
False
>>> # The config can be also visited by dict form, the following code also works
>>> print(fi_model.conv1.FI_config['activation'][0]['enabled'])
False
>>> 
>>> # Show and modify fault injection rate of resnet18.conv1
>>> print(fi_model.conv1.FI_config.activation[0].selector.args.rate)
'1e-3'
>>> fi_model.conv1.FI_config.activation[0].selector.args.rate = 1e-5
>>> print(fi_model.conv1.FI_config.activation[0].selector.args.rate)
1e-05
>>> 
>>> # Now different layer has different config, let's print
>>> pprint(fi_model.conv1.FI_config.activation[0].state_dict())
{'enabled': False,
 'error_mode': {'args': {'bit_width': 16}, 'method': 'IntSignBitFlip'},
 'quantization': {'args': {'bit_width': 16, 'dynamic_range': 'auto'},
                  'layerwise': True,
                  'method': 'SymmericQuantization'},
 'selector': {'args': {'poisson': True, 'rate': 1e-05},
              'method': 'RandomPositionByRate'}}
>>> # print configuration of any other layer
>>> pprint(fi_model.layer1[0].conv1.FI_config.activation[0].state_dict())
{'enabled': True,
 'error_mode': {'args': {'bit_width': 16}, 'method': 'IntSignBitFlip'},
 'quantization': {'args': {'bit_width': 16, 'dynamic_range': 'auto'},
                  'layerwise': True,
                  'method': 'SymmericQuantization'},
 'selector': {'args': {'poisson': True, 'rate': '1e-3'},
              'method': 'RandomPositionByRate'}}
```

Note that MRFI dynamically executes the content in ConfigTree, 
which means that when using `fi_model` later in the code above, 
the conv1 layer and other layers will execute fault injection in different configurations.

## Batched modify

Sometimes we need to modify a set of model parameters, such as the error rate on all convolutional layers.

Firstly, we may visit every config of module by one loop using PyTorch's `named_modules()`.

```python title="use PyTorch's function to visit each module"
>>> for name, layer in fi_model.named_modules():
...     if 'conv' in name:
...         print(name, layer.FI_config.activation[0].selector.args.rate) # 1e-3 except conv1
...         layer.FI_config.activation[0].selector.args.rate = 1e-5 # then set to 1e-5
...
conv1 1e-05
layer1.0.conv1 1e-3
layer1.0.conv2 1e-3
layer1.1.conv1 1e-3
layer1.1.conv2 1e-3
layer2.0.conv1 1e-3
layer2.0.conv2 1e-3
layer2.1.conv1 1e-3
layer2.1.conv2 1e-3
layer3.0.conv1 1e-3
layer3.0.conv2 1e-3
layer3.1.conv1 1e-3
layer3.1.conv2 1e-3
layer4.0.conv1 1e-3
layer4.0.conv2 1e-3
layer4.1.conv1 1e-3
layer4.1.conv2 1e-3
```

This allows us to conduct some selective protection experiments through a for loop, 
with only set the `enabled` True or False for some layers.

However, the above code may be complex, therefore, 
MRFI provides a more convenient way to access a batch of parameters. They are 
`fi_model.get_activation_configs()` and `fi_model.get_weights_configs()`.

```python title="Modify a batch of config parameter using MRFI's API"
>>> selector_args = fi_model.get_activation_configs('selector.args', module_name = 'conv')
>>> 
>>> # Previously, we set 1e-5 using the for loop, let's print
>>> print(selector_args.rate)
[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05]
>>> selector_args.rate = 1e-3
>>> print(selector_args.rate)
[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
```
Nice, we can set injection rate on all convolution layers.
This allow us to conduct fault injection on different bit error rate.

It is worth mentioning that `mrfi.experiment.BER_Acc_experiment` works in this way, 
as long as a list of selector configuration nodes is provided, 
it can automatically complete a batch of experiments with different injection rates.

We can also set a block with 4 layers in ResNet18 with different injection rate.

```python
>>> # match config of block1 in ResNet18
>>> block1_selector_args = fi_model.get_activation_configs('selector.args', module_name = 'layer1')
>>>
>>> print(block1_selector_args.rate)
[0.001, 0.001, 0.001, 0.001]
>>>
>>> # We can set block1 of ResNet18 to a list of different injection rate.
>>> block1_selector_args.rate = [1e-8, 1e-7, 1e-6, 1e-5]
>>> print(selector_args.rate) # print rate of all conv layers now
[0.001, 1e-08, 1e-07, 1e-06, 1e-05, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
>>>
>>> # However, if we set a list of config node to one scalar, e.g. 1e-6, 
>>> # this will be broadcasted onto this list.
>>> block1_selector.rate = 1e-6
>>> print(selector_args.rate)
[0.001, 1e-06, 1e-06, 1e-06, 1e-06, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
>>>
```

By this way, we can set a batch of any other configuration(e.g. scale factor of quantization), too.

See more about how to these two functions in [`MRFI.get_configs()`](/mrfi/#mrfi.mrfi.MRFI.get_configs).

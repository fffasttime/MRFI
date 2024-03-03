# Basic usage

## Beginning

As a starting point, you can give a **PyTorch model** (`torch.nn.Module` type) and a **EasyConfig** to MRFI to create a model with fault injection.

```python
from mrfi import MRFI, EasyConfig
from mymodel import Model  # A custom model
from torchvision.models import resnet18  # A model from torchvision

fi_model = MRFI(Model(), EasyConfig.load_file('easyconfigs/default_fi.yaml'))
fi_resnet18 = MRFI(resnet18(), EasyConfig.load_preset('default_fi'))
```

Then, you can use `fi_model` or `fi_resnet18` as a regular PyTorch model for inference, 
and the output of model will be affected by fault injection. 
More exactly, the hooks inserted by MRFI will conduct fault injection automatically according to the configuration in **EasyConfig**.

To test accuracy of a classification model, you may write a loop to compare the outputs and labels. 
For convenience, we also provide some utility functions in `mrfi.experiment` that can do it in one line, such as `Acc_experiment`.

See complete exmpale of [basic fault injection on LeNet](basic_faultinjection.md).

### EasyConfig usage

We provide some template of easyconfig file in folder `easyconfigs/`. 
You can modify these configuration files according to your needs and load it by  `EasyConfig.load_file()`.

```yaml title="easyconfigs/default_fi.yaml"
faultinject:
  - type: activation
    quantization:
      method: SymmericQuantization
      dynamic_range: auto
      bit_width: 16
    enabled: True
    selector:
      method: RandomPositionByRate
      poisson: True
      rate: 1e-3
    error_mode:
      method: IntSignBitFlip
      bit_width: 16

    module_name: [conv, fc] # Match layers that contain these name
    module_type: [Conv2d, Linear] # Or match layers contain these typename
```
It is clear the quantization, selector, error_mode method used for fault injection, as well as their parameters. 
The order of these config blocks in same level is not important because YAML regards them as dictionary.

Let's look one parameter. `rate` is a parameter of selector `RandomPositionByRate` indicate the indicate the probability of each tensor value being selected.
Here, we set  `rate: 1e-3` to significantly indicate that error injection is working, which is a relative high error rate for fault injection.

You can use the commonly used methods already provided by MRFI as the execution module(i.e. observer, selector, error_mode or quantization), 
see all methods in [MRFI function table](function_table.md). 
You can also customize new methods based on their interfaces. See [Custom Method/Arguments in MRFI](usage_custom.md).

The last two lines indicate the layers that require fault injection. 
MRFI will try to match all layers specified by `module_name` or `module_type` and set fault injection above.
Before write a custom fault injection in your own model, 
You can type `print(model)` to get name and type of all layers of the model.

If you don't want to create a new YAML EasyConfig file,
you can also load a custom EasyConfig yaml from python string by `EasyConfig.load_string()`,
or modify EasyConfig object by python code directly before create a MRFI object.

See [EasyConfig Usage](usage_easyconfig.md) to learn more usage and special rules of EasyConfig,

### Observing variables

It is helpful to make some observation before fault injection, such as the shape of tensors and the min-max range of tensors.
When conducting fault injection, internal observers are also helpful for recognize error propagation.

In MRFI, observers can be defined in **EasyConfig**, then they will be called every inference. 
You can collect observation results by using `fi_model.observers_result()` after each run.

For convenience, `mrfi.experiment` provides some function for inserting observer temporarily and making inference.
`get_activation_info` and `get_weight_info` in `mrfi.experiment` are powerful, allow us to perform various observations in one line of code.
They can be used to observe the shape, distribution, and sampling & visualization of network data.

See [example of basic observe on LeNet with interactive command](basic_observe.md).

### Golden run

To inference a `fi_model` without fault injection, we suggest using a golden_run context to include the code that calls the model,
which will temporary disable all fault injection on the model:

```python
with fi_model.golden_run():
    # Disable MRFI in this context
    out_golden = fi_model(input_data)

out_fi = fi_model(input_data)

# Or disable MRFI directly
fi_model.golden = True
out_golden = fi_model(input_data)
```

### Fine-grained and advanced configuration

MRFI allows different error injection methods and parameters to be configured on each layer, as it has a tree like detailed configuration **ConfigTree** inside. 
*In fact, when we created MRFI objects from EasyConfig earlier, the configuration was automatically copied to all matching layers and modules to create ConfigTree.*

To execute different error injection parameters on different layers, one way is to write several injection configuration in **EasyConfig** and let MRFI to expand them to distinct layers.
However, a more flexible approach is to directly modify **ConfigTree**. 

It is possible to manually modify the ConfigTree YAML configuration file and then load it. 
Due to the large amount of data, we also provide some APIs for dynamically modifying a batch of configurations of the built-in ConfigTree.

See [fine-grained configuration](usage_finegrained.md) and [advanced configuration](usage_advanced.md) for more information.

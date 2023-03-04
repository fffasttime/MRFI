# Multi-Resolution Fault Injector

This is a neuron level fault injector based on PyTorch. 

Compared with other injection frameworks, the biggest feature is that it can flexibly adjust different injection configurations for different experimental needs. Injection config and observations on each layer can be set independently by one clear config file.

When we verify our statistic model based analysis of reliability, we found that the previous framework is not easy to be used for such experiments that require highly flexible configuration. We developed this fault injector for complex experimental requirements.

This project needs more development, because some of its functions are not easy to use. 

## Example

This is a example experiment code for LeNet. 

### Fault inject config example

This is a example yaml config for LeNet. The configuration file is easy to read and flexible.

```yaml
FI_activation: true
FI_enable: false
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 8
layerwise_quantization:
  bit_width: 8
  dynamic_range: 8
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0008
sub_modules:
  conv1:
    layerwise_quantization:
      bit_width: 8
      dynamic_range: 4
    observer:
      map: mse
      reduce: sum
  conv2:
    observer:
      map: mse
      reduce: sum
  fc1:
    observer:
      map: mse
      reduce: sum
  fc2:
    observer:
      map: mse
      reduce: sum
  fc3:
    observer:
      map: mse
      reduce: sum
```

In this configuration, error injection is only performed on the active value (`FI_activation: true`).

The fault inject mode is most significant bit of integer value (`flip_mode: flip_int_highest`).

`selector` parameter specify how to select victim neurons, it is random choiced from all neurons of rate 8e-4 here.

Next we can choose layers to be fault injected in `sub_modules`. These layer name is the same as the module name when created in PyTorch.

We can specify FI parameters of each layer. For example, we set conv1 with smaller dynamic_range here. Note that FI parameters is "inherit" from parent level, so we can omit the configuration for each layer and let their flip_mode and selector parameter same as we specified in model level.

Some times we need some internal observation data of the model, e.g. dynamic range of activations or measurement of error perturbation. We can simply add different observer into layers. For example, we can measure error perturbation by calculate mean square error (`map: mse`) of the layer output, and then sum them along different input execution (`reduce: sum`). 

The `map` here means how to map a layer output into a result. If we want to find the dynamic range of each layer, we can set (`map: maxabs`) to get max range of one batch layer output, and then reduce them by (`reduce: max`) to get max range between all batchs.


### python script for per layer fault tolerance comparing

Previous configuration file gives us a static fault inject config, and error injection is performed on all five layers of lenet with a fixed probability.

Assume we want to conduct fault inject experiment for each single layer so that we can compare the fault tolerance ability of each layer. In this case, we have to perform 5 experiments and enable only one layer for inject in each experiment. To run these highly similar experiments, it is awkward to copy many configuration file or rewrite config file by script code.

In MRFI, we can enable or disable fault injection of each layer dynamically, so it can be easily completed in a loop. You can refer the following code. Note that you can also dynamically modify other arguments as your need, such as quantization dynamic range or fault inject rate.

```python
import yaml
import torch
import numpy as np

from model.vgg11 import Net, testset
from mrfi.injector import ModuleInjector

def experiment(total = 10000): # total = number of images to test
    torch.set_num_threads(8)
    config = yaml.load(yamlcfg)

    net=Net(True) # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False) # cifar dataloader

    FI_network = ModuleInjector(net, config) # create FI framework

    for layer in FI_network.subinjectors: # select each layer for fault inject
        data=iter(testloader)
        FI_network.reset_observe_value() # reset internal observation value so that different experiment result WILL NOT be wrongly accumulated
        
        accg, acc = 0, 0
        for ll in FI_network.subinjectors: # disable other layer fault injection
            ll.FI_enable = False
        layer.FI_enable = True # enable current layer injection

        for i in range(total): # input images
            images, labels = next(data)
            images=images.to(device)

            outg = FI_network(images, golden=True) # Golden run before FI run is necessary.
            out = FI_network(images).cpu().numpy() # FI run. Note that neural network internal observation value will be record automatically.

            accg+=(np.argmax(outg[0])==labels.numpy()[0]) # sum final accuracy, similar with normal accuracy test
            acc+=(np.argmax(out[0])==labels.numpy()[0])

        observes=FI_network.get_observes()
        print(layer.name, end='\t')
        for name, value in observes.items():   # print internal observation values, it will print RMSE of each layer activation value if we use previous yaml config.
            print("%.5f"%np.sqrt(value/total), end='\t')
        print("%.2f%%"%(acc/total*100), flush=True, end='\t') # print FI accuracy
        print("%.2f%%"%(accg/total*100), flush=True)          # print golden accuracy
```

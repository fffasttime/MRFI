# Multi-Resolution Fault Injector

<p dir="auto" align="center">
<img src="docs/assets/logo_name.png" alt="MRFI Logo" width=350)>
</p>

## MRFI Overview
[![GitHub license](https://img.shields.io/github/license/fffasttime/MRFI)](https://github.com/fffasttime/MRFI/blob/master/LICENSE)
[![Test](https://github.com/fffasttime/MRFI/actions/workflows/codecov.yml/badge.svg)](https://github.com/fffasttime/MRFI/actions/workflows/codecov.yml)
[![Codecov](https://codecov.io/gh/fffasttime/MRFI/branch/main/graph/badge.svg)](https://codecov.io/gh/fffasttime/MRFI)
[![PyPI](https://img.shields.io/pypi/v/MRFI)](https://pypi.org/project/MRFI/)

Multi-Resolution Fault Injector is a **powerful** neural network fault injector tool based on PyTorch.

Compared with previous network-level fault injection frameworks, the biggest feature is that MRFI can flexibly adjust different injection configurations for different experimental needs. 
Injection config and observations on each layer can be set **independently** by one clear config file. 
MRFI also provides a large number of commonly used fault injection methods and error models, and allows customization.
In preliminary experiments, we may not want to face complex experimental configurations. MRFI also provide simple API for course-grained fault injection experiment.

Read detail usage on [MRFI Documents >>](https://fffasttime.github.io/MRFI/)

![Overview Pic](/docs/assets/overviewpic.png)

Fault injection is an important classic method for studying reliability. 
This is crucial for the design and deployment of neural networks for security critical applications.
The low-level hardware based fault injection methods are time-consuming, 
while the high-level network model based methods usually neglect details.
Therefore, MRFI is designed to be an efficient and highly configurable multi-resolution network-level fault injection tool.

Learn more from our [paper of MRFI on Arxiv](https://arxiv.org/pdf/2306.11758.pdf). For other level fault injection tool, see also [Neural-Network-Reliability-Analysis-Toolbox](https://github.com/fffasttime/Neural-Network-Reliability-Analysis-Toolbox).

## Basic Example

### Install MRFI

```bash
# Install by pip
pip install MRFI

# Or download from github
git clone https://github.com/fffasttime/MRFI.git
pip install -r requirements.txt
```

### A coarse-grained configuation fault inject experiment
The following code perform a quantized random integer bit flip injection on LeNet, 
and find the relation between bit error rate (BER) and classification accuracy.

```python title="LeNet default fault injection"
from dataset.lenet_cifar import make_testloader, LeNet
from mrfi import MRFI, EasyConfig
from mrfi.experiment import Acc_experiment, Acc_golden, BER_Acc_experiment

testloader = make_testloader(1000, batch_size = 128) # test on 1000 cifar-10 images

# Create fault inject model
fi_model = MRFI(LeNet(trained=True).eval(), 
                EasyConfig.load_preset('default_fi'))

################### A Simple acccuracy experiment #####################
# Test accuracy under fault injection with select rate = 1e-3 which specified in "default_fi"
print('FI Acc: ', Acc_experiment(fi_model, testloader))
# Test accuracy w/o fault inject
with fi_model.golden_run():
    print('golden run Acc: ', Acc_experiment(fi_model, testloader))

######################## BER_Acc_experiment ###########################
# Get selector handler because BER_Acc_experiment needs to modify selection rate in experiment
selector_cfg = fi_model.get_activation_configs('selector')
BER, Acc = BER_Acc_experiment(fi_model, selector_cfg, 
                              make_testloader(1000, batch_size = 128), 
                              [1e-6, 1e-5, 1e-4, 1e-3])
print('Bit error rate and accuracy: ', BER, Acc)
```

A possible output should like this:
> FI Acc:  0.597
> 
> golden run Acc:  0.638
>
> Bit error rate and accuracy:  [1.e-06 1.e-05 1.e-04 1.e-03] [0.637 0.624 0.539 0.247]

The content of corresponding EasyConfig file `default_fi.yaml`:
```yaml
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

For detail usage, see [MRFI Doc - Usage](https://fffasttime.github.io/MRFI/usage/).

Explore MRFI internal functions in [MRFI Doc - function table](https://fffasttime.github.io/MRFI/function_table/).

## Supported Features

**Activation injection**

- [x] Fixed position (Permanent fault)
- [x] Runtime random position (Transient fault)

**Weight injection**

- [x] Fixed position (Permanent fault)
- [x] Runtime random position (Transient fault)

**Injection on quantization model**

- [x] Posting training quantization
- [x] Dynamic quantization
- [x] Fine-grained quantization parameters config
- [x] Add custom quantization

**Error mode**

- [x] Integer bit flip
- [x] Float bit flip
- [x] Stuck-at fault (SetValue)
- [x] Random value
- [x] Add random noise

**Internal observation & visualize**

- [x] Activation & Weight observer
- [x] Error propagation observer
- [x] Easy to save and visualize result, work well with `numpy` and `matplotlib`

**Flexibility**

- [x] Add custom `error_mode`, `selector`, `quantization` and `observer`
- [x] Distinguish network-level, layer-level, channel-level, neuron-level and bit-level fault tolerance difference

**Performance**

- [x] Automatically use GPU for network inference and fault injection
- [x] The selector - injector design is significantly faster than generate probability on all position when perform a random error injection
- [x] Accelerate error impact analysis through internal observer metrics rather than use original accuracy metric

**Fine-grained configuration**

- [x] By python code
- [x] By .yaml config file
- [ ] By GUI

**Evaluation fault tolerance policy**

- [x] Selective protection on different level
- [ ] More fault tolerance method may be support later (e.g. fault tolerant retrain, range-based filter)


## Developing

Code in `mrfi/` are tested to ensure correctness. Test case are under `test/`.

Set `logging.basicConfig(level = logging.DEBUG)` can get more runtime logging. It may be helpful when MRFI does not perform as expected.

### Unit Test

```bash
pytest --cov=mrfi --cov-report=html
```
coverage report written to `htmlconv/index.html`

### Update Docs

Modify markdown content files under `docs/`.
```bash
mkdocs build
mkdocs serve
mkdocs gh-deploy
```

### Bug Report

I tried to conduct thorough testing for MRFI, but due to the many features, there may still be potential issues. 
If you find a problem, please provide sufficient code and configuration in the issue to analyze the cause, or submit a pull request to fix the problem.

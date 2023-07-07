# Multi-Resolution Fault Injector

<p dir="auto" align="center">
<img src="docs/assets/logo_name.png" width=350)>
</p>

Multi-Resolution Fault Injector is a **powerful** neural network fault injector tool based on PyTorch.

Compared with other injection frameworks, the biggest feature is that it can flexibly adjust different injection configurations for different experimental needs. Injection config and observations on each layer can be set independently by one clear config file. MRFI also provides a large number of commonly used error injection methods and error models, and allows customization.

![Overview Pic](/docs/assets/overviewpic.png)

Read detail usage from Documents >>

Learn more from our paper on [Arxiv](https://arxiv.org/pdf/2306.11758.pdf).

## Basic Example

### A coarse-grained configuation fault inject experiment
The following code perform a quantized random integer bit flip injection on LeNet, 
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
- [ ] By GUI (in development)

**Evaluation fault tolerance policy**

- [x] Selective protection on different level
- [ ] More fault tolerance method may be support later (e.g. fault tolerant retrain, range-based filter)



## Developing

Code in `mrfi/` are tested to ensure correctness. Test case are under `test/`.

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

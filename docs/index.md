# MRFI

## Overview

Multi-Resolution Fault Injector is a powerful neural network fault injector based on PyTorch.

<p dir="auto" align="center">
<img src="assets/logo_name.png" width=350)>
</p>

Compared with other injection frameworks, the biggest feature is that it can flexibly adjust different injection configurations for different experimental needs. Injection config and observations on each layer can be set independently by one clear config file. MRFI also provides a large number of commonly used error injection methods and error models, and allows customization.

![Overview Pic](/assets/overviewpic.png)

In preliminary experiments, you may not want to face complex experimental configurations. For example, simply observing the parameters of the network model, or conducting error injection experiments with a simple global configuration. MRFI also provide simple API for observation and course-grained fault injection.

[>>Examples for basic usage](usage.md) 

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

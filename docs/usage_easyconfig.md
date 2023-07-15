# EasyConfig usage

## EasyConfig YAML format

YAML is a human readable format that uses white space indentation to represent levels and rows to represent list and dictionary items.
See [yaml.org](https://yaml.org).

### Full EasyConfig YAML template


```yaml title="full easyconfig template"
faultinject:
  - type: activation
    enabled: True # optional, default is True
    quantization: # optional
      method: SymmericQuantization
      dynamic_range: auto
      layerwise: True
      bit_width: 16
    selector: # optional. If not specified, all value will be selected
      method: RandomPositionByRate
      poisson: True # optional because it's a optional argument of method RandomPositionByRate
      rate: 1e-3
    error_mode:
      method: IntSignBitFlip
      bit_width: 16

    module_name: conv # Match layers that contain these name
    module_type: [Conv2d, Linear] # Or match layers contain these typename
    module_fullname: [fc1, fc2, layer1.0.conv1] # Or do exact match

  - type: weight
    weight: [weight, bias] # optional, default is "weight"

    error_mode:
      method: SetValue
      value: 0
    module_name: conv

oberve:
  - method: MinMax
    position: pre  # optional, default is "after"
    module_name: fc 
```

An easyconfg YAML file has 4 levels:

    1 The first level begin with `faultinject:` or `observe:`.
      2 List of injectors and observers definition, each block begin with `-`.
        3 Properties of fault injectors and observers, and execution modules(quantization, selector, error_mode).    
          4 Properties and parameters of execution modules.

### Injector properties

Each fault injector definition(level 3) can have following properties:

|name|value|description|
|-|-|-|
|type|one of `"activation_in"`,`"activation_out"` or `"weight"`|Which place will the fault injection to be performed. Set `activation` stands for `activation_in`.|
|enabled|optional, default is `True`|Enable this fault injection by default.|
|quantization|`Dict`of properties|Optional. If not specified, Will not make quantization.|
|selector|`Dict`of properties|Optional. If not specified, all tensor values will be selected, which may be slow for large tensors.|
|error_mode|`Dict`of properties||
|module_name|`str` or `list[str]`|Match layer or module by name. For example, layer "block1.conv1" will be matched by str "conv".|
|module_type|`str` or `list[str]`|Match layer or module by name. For example, layer with typename "Conv2d" will be matched by str "Conv".|
|module_fullname|`str` or `list[str]`|Select layer or module by full name. Layer "block1.conv1" won't be matched by str "conv1", only full name work.|
|weight|optional, `str` or `list[str]`, default is `"weight"`|When fault injection on weight, specify which weight name will be selected to inject.|

### Injector execution module properties

Each execution module properties(level 4) have "method" field. Other parameter fields are corresponding to each method.
For example, `RandomPositionByRate` is a selecter function defined in `mrfi.selector`, 
whose signature is `RandomPositionByRate(shape, rate=0.0001, poisson=True)`. 
We can set `rate: 1e-3` here in easyconfig. 
The optional argument can be emitted and use the default value in function definition.

Special: "layerwise" is optional field in quantization indicate MRFI to quantize the full tensor, default is True. 
If set to False, only the positions selected by selector will be quantized.

### Observer properties

Each observer definition(level 3) has "method", "position" field. 

"position" is a optional field that can be one of `pre` or `after`. 
This indicates whether MRFI observes values from the input or output of the layer.
Default is `after`.

The "module_name", "module_type", "module_fullname" field have the same meaning as injector properties.

## Python Class EasyConfig

[Class EasyConfig](../mrfi/#mrfi.mrfi.EasyConfig) is an editable dict-like object in Python.
That means you can modify it before give it to MRFI for expanding.

The following code modify a default EasyConfig without change a YAML file.

```python
econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')

# Set error_mode method to 'IntRandomBitFlip'.
econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
# 'set_error_mode()' can do the same thing
econfig.set_error_mode(0, {'method':'IntRandomBitFlip'}, True)

# Set layer and moudle to be matched
econfig.faultinject[0]['module_type'] = ['BatchNorm', 'Linear']
# 'set_module_used()' can do the same thing
econfig.set_module_used(0, module_type = ['BatchNorm', 'Linear'])

# Finally load it
fi_model = MRFI(Model(trained=True), econfig)
```


Method of EasyConfig:

::: mrfi.mrfi.EasyConfig
    options:
      show_root_heading: True
      show_source: False
      show_if_no_docstring: True
      show_docstring_description: False

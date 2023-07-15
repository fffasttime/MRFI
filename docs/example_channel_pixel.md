# Channel/Pixel experiment

The following code segments conduct channelwise and pixelwise injection on VGGNet-11.
The selector method used are `SelectedDimRandomPositionByNumber` and `FixedPixelByNumber`.

We select some layers for experimet and conduct experiment along all channel/pixel, 
so this will be a bit time-consuming.

The visualized result can be seen in our paper.

## Channelwise injection

```python
econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
econfig.set_module_used(0, 'Conv')
econfig.set_selector(0, {'method':'SelectedDimRandomPositionByNumber', 'n':batch_size * 10, 'channel': [0]})

fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), econfig)
selector_cfg = fi_model.get_activation_configs('selector')
action_cfg = fi_model.get_activation_configs()

action_cfg.enabled = False

shapes = get_activation_info(fi_model, make_testloader(1), method='Shape', module_type = 'Conv', pre_hook = True)
print(shapes)
shapes = list(shapes.values())

results = {}

for layer in range(1,4):
    action_cfg.enabled = False
    action_cfg[layer].enabled = True
    channels = shapes[layer][1]
    rmses = np.empty((channels,))

    for i in range(channels):
        selector_cfg[layer].args.channel = [i]
        rmse = observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), module_fullname = '')
        rmse = list(rmse.values())[0]
        print(layer, i, rmse)
        rmses[i] = rmse

    results['conv%d'%layer] = rmses

np.savez('result/vgg11_channelwise.npz', **results)
```

## Pixelwise injection

```python
config = """
faultinject:
  - type: activation_out
    quantization:
      method: SymmericQuantization
      dynamic_range: auto
      bit_width: 16
    enabled: True
    selector:
      method: FixedPixelByNumber
      pixel: (0,0)
      n: 10
      per_instance: True
    error_mode:
      method: IntSignBitFlip
      bit_width: 16

    module_name: features
"""

fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), EasyConfig.load_string(config))
selector_cfg = fi_model.get_activation_configs('selector', out=True)
activation_cfg = fi_model.get_activation_configs(is_out=True)

activation_cfg.enabled = False

shapes = get_activation_info(fi_model, make_testloader(1), method='Shape', module_type = 'Conv')
print(shapes)
shapes = list(shapes.values())

results = {}

for layer in [3, 5]:
    activation_cfg.enabled = False
    activation_cfg[layer].enabled = True
    print(shapes[layer])
    pixels = shapes[layer][2], shapes[layer][3]
    rmses = np.empty(pixels)

    for i in range(pixels[0]):
        for j in range(pixels[1]):
          selector_cfg[layer].args.pixel = (i, j)
          rmse = observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), module_fullname = '')
          rmse = list(rmse.values())[0]
          print(layer, i, j, rmse)
          rmses[i, j] = rmse

    results['conv%d'%layer] = rmses

np.savez('result/vgg11_pixelwise_o.npz', **results)
```
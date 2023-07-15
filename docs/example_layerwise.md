# Layerwise experiment

The complete code below provides a layerwise fault injection in order to distinguish fault tolerance difference among layer.

With the dynamic fine-grained configuration ability of MRFI introduced in [advanced usage](usage_advanced.md),
we can set per layer fault injection enabled or disabled by one loop, without modify ConfigTree manually.

Additionally, if the GPU is available, the following code automatically uses CUDA for inference and fault injection.

```python
import torch
from dataset.lenet_cifar import testset, Net
from mrfi import MRFI, EasyConfig
import sys

configfilename = 'weight_fi'
if len(sys.argv)>1:
    configfilename = sys.argv[1]

fi_model = MRFI(Net(trained=True), EasyConfig.load_file('easyconfigs/%s.yaml'%configfilename))
fi_model.save_config('detailconfigs/lenet_%s.yaml'%configfilename)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    fi_model.cuda()
else:
    device = torch.device('cpu')

# method 1
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

cfg = fi_model.get_weights_configs()
print(cfg)

for i in range(5): # 5 layers in lenet
    acc_golden, acc_fi = 0, 0
    cfg.enabled = False
    cfg[i].enabled = True
    fi_model.observers_reset()
    for images, labels in testloader:
        images=images.to(device)
        
        # fault free run
        with fi_model.golden_run():
            outs_golden=fi_model(images)
        # fault inject run
        outs_fi=fi_model(images)

        acc_golden+=(outs_golden.argmax(1)==labels).sum().item()
        acc_fi+=(outs_fi.argmax(1)==labels).sum().item()

    print(f'{len(testset)} images, acc_golden {acc_golden}, acc_fi {acc_fi}')
    print(fi_model.observers_result())
```

Possible output:

```
[<'FI_ATTR' node 'model.sub_modules.conv1.weights.0'>, <'FI_ATTR' node 'model.sub_modules.conv2.weights.0'>, <'FI_ATTR' node 'model.sub_modules.fc1.weights.0'>, <'FI_ATTR' node 'model.sub_modules.fc2.weights.0'>, <'FI_ATTR' node 'model.sub_modules.fc3.weights.0'>]
10000 images, acc_golden 6226, acc_fi 6189
{}
10000 images, acc_golden 6033, acc_fi 5981
{}
10000 images, acc_golden 5974, acc_fi 5906
{}
10000 images, acc_golden 5965, acc_fi 5954
{}
10000 images, acc_golden 5961, acc_fi 5937
{}
```

The `observers_result` dictionary is empty because we did not set a observer.

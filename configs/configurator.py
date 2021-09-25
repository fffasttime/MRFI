import yaml
import os
from copy import deepcopy

default_observer_config={
    'map':'mse',
    'reduce':'sum',
}

default_config={
    'FI_enable': True,
    'FI_weight': False,
    'FI_activation': True,
    'selector': 'RandomPositionSelector_Rate', 
    'selector_args': {
        'rate': 1e-4,
        'poisson': True
    },
    'layerwise_quantization': {'bit_width': 8, 'bound':'auto'},
    'flip_mode': 'flip_int_hightest',
    'flip_mode_args': {'bit_width': 8},
    'sub_modules': dict(),
    'observer': default_observer_config,
}

class Configurator:
    def __init__(self, network, module_name, expand=0):
        self.config = deepcopy(default_config)
        self.module_name = module_name
        self.load_sub_modules(self.config, network, expand)

    def save(self, filename):
        with open('configs/'+filename, 'w') as f:
            print(yaml.dump(self.config), file=f)
        print(yaml.dump(self.config))

    def load_sub_modules(self, cur_config, cur_module, expand):
        for sub_module_name, sub_module in cur_module.named_children():
            if any([name in sub_module_name for name in self.module_name]):
                if expand>0:
                    cur_config['sub_modules'][sub_module_name] = deepcopy(default_config)
                else:
                    cur_config['sub_modules'][sub_module_name] = {}
                self.load_sub_modules(cur_config['sub_modules'][sub_module_name], sub_module, expand-1)

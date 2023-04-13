from typing import Callable
from numpy import mod
import torch.nn as nn
import torch
from .selector import Selector_Dict
from .error_mode import FlipMode_Dict
from .observer import Mapper_Dict, Reducer_Dict

import logging
logger = logging.getLogger('mrfi')

def layerwise_quantization(x, bit_width, dynamic_range):
    down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
    x += dynamic_range
    x *= (up_limit - down_limit) / (dynamic_range*2)
    x += down_limit
    x.clamp_(down_limit, up_limit)
    x.round_()

def layerwise_dequantization(x, bit_width, dynamic_range):
    down_limit, up_limit = -(1<<bit_width-1)+1, (1<<bit_width-1)-1
    x -= down_limit
    x *= (dynamic_range*2)/(up_limit - down_limit)
    x -= dynamic_range

def layerwise_quantization_pos(x, bit_width, dynamic_range):
    up_limit = (1<<bit_width)-1
    x *= (up_limit) / (dynamic_range)
    x.clamp_(0, up_limit)
    x.round_()

def layerwise_dequantization_pos(x, bit_width, dynamic_range):
    up_limit = (1<<bit_width)-1
    x *= (dynamic_range)/(up_limit)

def input_hook_func(self, module, input):
    input, = input  # unpack
    if self.FI_enable and self.FI_activation and self.layerwise_quantization:
        if self.layerwise_quantization_dynamic_range is None:
            dynamic_range = input.abs().max()
        else:
            dynamic_range = self.layerwise_quantization_dynamic_range
        layerwise_quantization(input, self.layerwise_quantization_bit_width, dynamic_range)
    
    if self.FI_enable and self.FI_activation and not self.root.record_golden:
        error_list = torch.as_tensor(self.selector.gen_list(input.shape))
        if error_list.numel()>0:
            values = input.view(-1)[error_list]
            input.view(-1)[error_list] = self.flip_mode(values, **self.flip_mode_args)

    if self.FI_enable and self.FI_activation and self.layerwise_quantization:
        layerwise_dequantization(input, self.layerwise_quantization_bit_width, dynamic_range)
    
    if self.FI_enable and self.FI_weight:  # get original weight
        if self.weight_original is None:  # fisrt run, backup
            if self.layerwise_quantization_dynamic_range_weight is None:
                self.dynamic_range_weight = module.weight.abs().max()
            else:
                self.dynamic_range_weight = self.layerwise_quantization_dynamic_range_weight
            logger.info(self.name + ' weight quantization range %f'%self.dynamic_range_weight)
            if self.layerwise_quantization:
                layerwise_quantization(module.weight, self.layerwise_quantization_bit_width, self.dynamic_range_weight)
            self.weight_original = module.weight.clone()
        else:
            module.weight.data = self.weight_original.clone()
            
        if not self.root.record_golden:
            error_list = torch.as_tensor(self.selector.gen_list(module.weight.shape))
            if error_list.numel()>0:
                values = module.weight.view(-1)[error_list]
                module.weight.view(-1)[error_list] = self.flip_mode(values, **self.flip_mode_args)
        if self.layerwise_quantization:
            layerwise_dequantization(module.weight, self.layerwise_quantization_bit_width, self.dynamic_range_weight)


def observer_hook_func(self, value: torch.Tensor):
    value = value.cpu().numpy().copy()
    if self.root.record_golden:
        self.golden_value = value
        return

    current_val = self.mapper(value.reshape(value.shape[0], -1), self.golden_value.reshape(self.golden_value.shape[0], -1))
    if self.observe_value is None:
        self.observe_value = self.reducer(current_val)
    else:
        self.observe_value = self.reducer([self.observe_value, self.reducer(current_val)])

class ModuleInjector:
    def __init__(self, module: nn.Module, config: dict, root_injector = None, name='network'):
        self.module = module
        self.subinjectors=[]
        self.name = name
        
        if root_injector is None: # root module
            root_injector = self
            self.record_golden = False
        
        self.root = root_injector
        if 'sub_modules' not in config:
            config['sub_modules']={}
        self.is_leaf_node = (len(config['sub_modules'])==0)
        
        self.__set_observer(module, config)
        if self.is_leaf_node:
            self.__set_FI(module, config)

        for (sub_module_name, sub_config) in config['sub_modules'].items():
            try:
                val=int(sub_module_name)
                sub_module=module[val]
            except ValueError:
                sub_module=getattr(module, sub_module_name)
            sub_injector = self.__set_sub_injector(config, sub_module, sub_config, str(sub_module_name))
            setattr(self, str(sub_module_name), sub_injector)

    def __set_FI(self, module, config):
        self.FI_enable = config.get('FI_enable', False)
        self.FI_activation = config.get('FI_activation', False)
        self.FI_weight = config.get('FI_weight', False)

        self.selector_name = config.get('selector', None)
        self.selector_args=config.get('selector_args', {})
        self.update_selector()

        self.flip_mode: Callable = FlipMode_Dict[config.get('flip_mode', None)]
        self.flip_mode_args=config.get('flip_mode_args', {})

        self.layerwise_quantization = False
        if 'layerwise_quantization' in config:
            self.layerwise_quantization = True
            self.layerwise_quantization_bit_width: int = config['layerwise_quantization']['bit_width']
            self.layerwise_quantization_dynamic_range = config['layerwise_quantization'].get('dynamic_range', None)
            self.layerwise_quantization_dynamic_range_weight = config['layerwise_quantization'].get('dynamic_range_weight', None)
            if self.layerwise_quantization_dynamic_range == 'auto':
                self.layerwise_quantization_dynamic_range = None
            if self.layerwise_quantization_dynamic_range_weight == 'auto':
                self.layerwise_quantization_dynamic_range_weight = None
        
        self.weight_original = None
        
        def input_hook_func_warp(mod, input):
            input_hook_func(self, module, input)

        self.input_hook = module.register_forward_pre_hook(input_hook_func_warp)
        logger.info("Add FI hook before " + self.name)
            
    def __set_observer(self, module: nn.Module, config):
        self.observe_value = None
        self.golden_value = None
        
        if 'observer' in config: 
            config = config['observer']
        else:
            return
        
        pre_hook = config.get('pre_hook', False)

        self.mapper: Callable = Mapper_Dict[config['map']]
        self.reducer: Callable = Reducer_Dict[config['reduce']]

        if pre_hook:
            self.observer_hook = module.register_forward_pre_hook(lambda mod, input: observer_hook_func(self, input[0]))
            logger.info("Add observation hook before " + self.name)
        else:
            self.observer_hook = module.register_forward_hook(lambda mod, input, output: observer_hook_func(self, output))
            logger.info("Add observation hook after " + self.name)


    def __set_sub_injector(self, cur_config, sub_module, sub_config, name):
        if sub_config is None:
            sub_config = {}
        # merge config (inherit from parent)
        for (key, val) in cur_config.items():
            if key == 'observer' or key == 'sub_modules':  # observer is not inherted
                continue
            if key not in sub_config:
                sub_config[key] = val
        
        subinjector = ModuleInjector(sub_module, sub_config, self.root, name)
        self.subinjectors.append(subinjector)
        return subinjector

    def update_selector(self):
        self.selector = Selector_Dict[self.selector_name](**self.selector_args)

    def reset_observe_value(self):
        self.observe_value = None
        for subinjector in self.subinjectors:
            subinjector.reset_observe_value()

    def get_observes(self, parname=''):
        observes_dict = {}
        for subinjector in self.subinjectors:
            observes_dict.update(subinjector.get_observes(parname+str(self.name)+'.'))
        if not(self.observe_value is None):
            observes_dict[parname + self.name] = self.observe_value
        return observes_dict

    def __call__(self, x, golden=False):
        self.root.record_golden = golden
        with torch.no_grad():
            return self.module(x)

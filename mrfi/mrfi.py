"""Core module of MRFI."""

import copy
import logging
from contextlib import contextmanager
from enum import Enum
from importlib import import_module
from typing import List, Optional, Union, Callable, Any
import yaml

import torch
from torch import nn

named_functions = {}

mrfi_packages = ['.observer', '.quantization', '.selector', '.error_mode']

def load_package_functions(name: str) -> None:
    """Load a package of functions.
    
    Args:
        name: module name.
    """
    if name[0] == '.':
        mod = import_module(name, 'mrfi')
        for k, v in mod.__dict__.items():
            if callable(v) and v.__module__ == 'mrfi'+name:
                named_functions[k] = v
    else:
        mod = import_module(name)
        for k, v in mod.__dict__.items():
            if callable(v) and v.__module__ == name:
                named_functions[k] = v

for package_name in mrfi_packages:
    load_package_functions(package_name)

def add_function(name: str, function_object: Callable[..., Any]) -> None:
    """Add a custom function, may be a observer, quantization, selector or error_mode.
    
    Args:
        name: To be used in config file, `method` field of observers, quantization, selector, error_mode.
        function_object: A callable object consists with implementions of observers, quantization, selector, error_mode.
    Examples:
        >>> mrfi.add_function('my_empty_selector', lambda shape: [])
        Then you can use this empty selector by setting selector.method to 'my_empty_selector' in .yaml config file.
    """
    named_functions[name] = function_object

class FIConfig:
    """Base class of FIConfig."""

def _read_config(filename: str) -> dict:
    """Read config file."""
    extentname = filename.split('.')[-1]
    with open(filename) as f:
        if extentname == 'yml' or extentname=='yaml':
            config = yaml.full_load(f)
        else:
            raise NotImplementedError(extentname)
    return config

def _write_config(config: dict, filename: str) -> None:
    """Write config file."""
    extentname = filename.split('.')[-1]
    with open(filename, 'w') as f:
        if extentname == 'yml' or extentname=='yaml':
            yaml.dump(config, f)
        else:
            raise NotImplementedError(extentname)

class EasyConfig(FIConfig):
    """EasyConfig object.
    
    Properties:
        - `faultinject`: List of fault injectors.
        - `observe`: List of observers.
    """
    def __init__(self, config: Optional[dict] = None):
        if config is None: config = {}
        self.faultinject = config.get('faultinject', [])
        self.observe = config.get('observe', [])
    
    @classmethod
    def load_file(cls, filename: str):
        """Load EasyConfig from .yaml File"""
        return cls(_read_config(filename))
    
    @classmethod
    def load_string(cls, string: str):
        return cls(yaml.full_load(string))
    
    def set_quantization(self, idx: int, content_dict, update_only = False) -> None:
        if update_only:
            self.faultinject[idx]['quantization'].update(content_dict)
        else:
            self.faultinject[idx]['quantization'] = content_dict

    def set_selector(self, idx: int, content_dict, update_only = False) -> None:
        if update_only:
            self.faultinject[idx]['selector'].update(content_dict)
        else:
            self.faultinject[idx]['selector'] = content_dict

    def set_error_mode(self, idx: int, content_dict, update_only = False) -> None:
        if update_only:
            self.faultinject[idx]['error_mode'].update(content_dict)
        else:
            self.faultinject[idx]['error_mode'] = content_dict
    
    def set_module_used(self, idx: int,
                        module_type: Union[str, List[str]] = None,
                        module_name: Union[str, List[str]] = None,
                        module_fullname: Union[str, List[str]] = None) -> None:
        if module_type is not None:
            self.faultinject[idx]['module_type'] = module_type
        else:
            self.faultinject[idx].pop('module_type', None)
        if module_name is not None:
            self.faultinject[idx]['module_name'] = module_name
        else:
            self.faultinject[idx].pop('module_name', None)
        if module_fullname is not None:
            self.faultinject[idx]['module_fullname'] = module_fullname
        else:
            self.faultinject[idx].pop('module_fullname', None)

class ConfigTreeNodeType(Enum):
    """Config tree node types.
    
    Note:
        A config tree has recursion structure:

        - MODULE
            - FI_LIST (e.g. activation_in, weights)
                - FI_ATTR
                    - FI_STAGE (e.g. quantization, selector)
                        - METHOD_ARGS (e.g. dynamic_range, rate)
            - OBSERVER_LIST
                - OBSERVER_ATTR
            - MODULE_LIST (sub_modules)
                - MODULE
                    - FI_LIST
                    - ...
    """
    MODULE = 0
    FI_LIST = 2
    OBSERVER_LIST = 3
    FI_ATTR = 4
    FI_STAGE = 5
    METHOD_ARGS = 6
    MODULE_LIST = 7
    OBSERVER_ATTR = 8

def _node_step(nodetype, name):
    if nodetype == ConfigTreeNodeType.MODULE:
        if name == 'sub_modules':
            return ConfigTreeNodeType.MODULE_LIST
        if name in ('weights', 'activation', 'activation_out'):
            return ConfigTreeNodeType.FI_LIST
        if name in ('observers_pre', 'observers'):
            return ConfigTreeNodeType.OBSERVER_LIST
        raise ValueError(name)
    if nodetype == ConfigTreeNodeType.FI_ATTR:
        if name in ('quantization', 'selector', 'error_mode'):
            return ConfigTreeNodeType.FI_STAGE
        if name in ('enabled', 'name'):
            return None
        raise ValueError(name)
    if nodetype == ConfigTreeNodeType.FI_STAGE:
        if name == 'method':
            return None
        if name == 'args':
            return ConfigTreeNodeType.METHOD_ARGS
        raise ValueError(name)
    if nodetype == ConfigTreeNodeType.MODULE_LIST:
        return ConfigTreeNodeType.MODULE
    if nodetype == ConfigTreeNodeType.FI_LIST:
        return ConfigTreeNodeType.FI_ATTR
    if nodetype == ConfigTreeNodeType.OBSERVER_LIST:
        return ConfigTreeNodeType.OBSERVER_ATTR
    if nodetype == ConfigTreeNodeType.METHOD_ARGS:
        return None
    if nodetype == ConfigTreeNodeType.OBSERVER_ATTR:
        return None
    assert False, f'Invalid name {name} nodetype {nodetype}'

class ConfigTree(FIConfig):
    """Make a detail config tree.
    
    Detail config tree is a dict-like object.
    """
    def __init__(self, config_node: dict, mrfi: 'MRFI', nodetype = ConfigTreeNodeType.MODULE, name: str = 'model'):
        """

        Args:
            config_node: A dict with ConfigTreeNodeType structrue.
            mrfi: MRFI object
            name: config node name for visualize.
        """
        self.__dict__['mrfi'] = mrfi
        self.__dict__['nodetype'] = nodetype
        self.__dict__['name'] = name
        if isinstance(config_node, list):
            raise ValueError(f"In config {self.name}, expect dict, got list")
        if isinstance(config_node, dict):
            self.__dict__['raw_dict'] = {}
            for k, v in config_node.items():
                subtype = _node_step(nodetype, k)
                if subtype is not None:
                    self.raw_dict[k] = ConfigTree(v, mrfi, subtype, self.name + '.' + str(k))
                else:
                    self.raw_dict[k] = v

        self.__dict__['weight_copy'] = None
        self.__dict__['object'] = None

    def __contains__(self, key):
        return key in self.raw_dict

    def __getitem__(self, index):
        return self.raw_dict[index]
    
    def __setitem__(self, index, value):
        self.raw_dict[index] = value
    
    def __len__(self):
        return len(self.raw_dict)

    def __getattr__(self, name):
        if name == 'golden':
            return self.mrfi.golden
        if name == 'enabled':
            return self.raw_dict['enabled'] and self.mrfi.global_FI_enabled
        if name in self.raw_dict:
            return self.raw_dict[name]
        raise KeyError(f"{self} has no config named '{name}'")
    
    def __setattr__(self, name, value):
        if name in self.__dict__['raw_dict']:
            self.__dict__['raw_dict'][name] = value
            return
        raise KeyError(f"{self} has no config named '{name}'")
    
    def __iter__(self):
        return iter(self.raw_dict.values())
    
    def __repr__(self) -> str:
        return f"<'{self.nodetype.name}' node '{self.name}'>"

    def append(self, obj) -> int:
        """Append a object on a list node"""
        assert self.nodetype in (ConfigTreeNodeType.MODULE_LIST, ConfigTreeNodeType.OBSERVER_LIST, ConfigTreeNodeType.FI_LIST)
        index = len(self.raw_dict)
        self.raw_dict[index] = obj
        return index
    
    def remove(self, index):
        """Remove object by index"""
        del self.raw_dict[index]
    
    def state_dict(self) -> dict:
        """Visualize or export config tree node and all sub node."""
        if self.nodetype == ConfigTreeNodeType.METHOD_ARGS:
            return self.raw_dict
        rdict = {}
        for k, v in self.raw_dict.items():
            if isinstance(v, ConfigTree):
                rdict[k] = v.state_dict()
            else:
                rdict[k] = v
        return rdict

    def hasattr(self, name) -> bool:
        """Determine if an attr is in config node."""
        return name in self.raw_dict
    
    def getattr(self, name):
        """Return an attr in config node, if not exist, return None"""
        if name not in self.raw_dict: return None
        return self.raw_dict[name]
    
    def set_internal_attr(self, key, name):
        """`__setattr__` reject bind new attr, use this if necessary."""
        self.__dict__[key] = name

def _FI_activation(config, act):
    """The core execution logic of activation value injection."""
    if not config.enabled: return
    
    layerwise_quantization, fi_quantization = False, False
    
    if config.hasattr('quantization'):
        if config.quantization.hasattr('layerwise') and config.quantization.layerwise == False:
            fi_quantization = True
        else:
            layerwise_quantization = True
    
    if layerwise_quantization or fi_quantization:
        quantization_method = named_functions[config.quantization.method]
        quantization_args = config.quantization.args.raw_dict
        if 'dynamic_range' in quantization_args:
            if (quantization_args['dynamic_range'] == 'max' or 
                quantization_args['dynamic_range'] == 'auto'):
                quantization_args = copy.copy(quantization_args)
                quantization_args['dynamic_range'] = act.abs().max()
    
    if layerwise_quantization:
        quantization_method.quantize(act, **quantization_args)
    
    if not config.golden:
        if config.hasattr('selector'):
            selector = named_functions[config.selector.method]
            selector_args = config.selector.args.raw_dict
            error_list = torch.as_tensor(selector(act.shape, **selector_args), dtype=torch.int64)
        else:
            error_list = slice(None)
        
        modifier = named_functions[config.error_mode.method]
        modifier_args = config.error_mode.args.raw_dict

        values = act.view(-1)[error_list]

        if fi_quantization: 
            quantization_method.quantize(values)

        fi_value = modifier(values, **modifier_args)
        
        if fi_quantization: 
            quantization_method.dequantize(fi_value)

        act.view(-1)[error_list] = fi_value

    if layerwise_quantization:
        quantization_method.dequantize(act, **quantization_args)

def _FI_activations(config, acts):
    """Fault inject on a series of activations."""
    if len(config)==0: return
    if isinstance(acts, tuple):
        assert (len(config)==1) or (len(config)==len(acts)), \
               'number of activation FI config should be 1 or same as module inputs'
        if len(config)==1:
            for act in acts:
                _FI_activation(config[0], act)
        else:
            for i, act in enumerate(acts):
                _FI_activation(config[i], act)
    else:
        _FI_activation(config[0], acts)

def _FI_weight(config, weight):
    """The core execution logic of weight value injection."""
    if not config.enabled: return
    
    layerwise_quantization, fi_quantization = False, False
    
    if config.hasattr('quantization'):
        if config.quantization.hasattr('layerwise') and config.quantization.layerwise == False:
            fi_quantization = True
        else:
            layerwise_quantization = True

    if config.weight_copy is None:
        config.set_internal_attr('weight_copy', weight.clone())
    else:
        weight.copy_(config.weight_copy)

    if layerwise_quantization or fi_quantization:
        quantization_method = named_functions[config.quantization.method]
        quantization_args = config.quantization.args.raw_dict
        if 'dynamic_range' in quantization_args:
            if (quantization_args['dynamic_range'] == 'max' or 
                quantization_args['dynamic_range'] == 'auto'):
                quantization_args = copy.copy(quantization_args)
                quantization_args['dynamic_range'] = weight.abs().max()
    
    if layerwise_quantization:
        quantization_method.quantize(weight, **quantization_args)
    
    if not config.golden:
        if config.hasattr('selector'):
            selector = named_functions[config.selector.method]
            selector_args = config.selector.args.raw_dict
            error_list = torch.as_tensor(selector(weight.shape, **selector_args), dtype=torch.int64)
        else:
            error_list = slice(None)
        
        modifier = named_functions[config.error_mode.method]
        modifier_args = config.error_mode.args.raw_dict

        values = weight.view(-1)[error_list]

        if fi_quantization: 
            quantization_method.quantize(values)

        fi_value = modifier(values, **modifier_args)
        
        if fi_quantization: 
            quantization_method.dequantize(fi_value)

        weight.view(-1)[error_list] = fi_value

    if layerwise_quantization:
        quantization_method.dequantize(weight, **quantization_args)

def _FI_weights(config, module):
    """Fault inject on a series of weights on a module."""
    if config is None: return
    for weight_config in config:
        if weight_config.enabled == False: # not None, True
            continue
        weight = getattr(module, weight_config.getattr('name'))
        _FI_weight(weight_config, weight)

def _run_observer(config, activation):
    if config.object is None:
        observer_method = named_functions[config.method]
        config.set_internal_attr('object', observer_method())

    if config.golden:
        config.object.update_golden(activation)
    else:
        config.object.update(activation)

def _run_observers(config, acts):
    """Run a series of observers on a module."""
    for observer in config:
        if isinstance(acts, tuple):
            index = int(observer.index) if observer.hasattr('index') else 0
            _run_observer(observer, acts[index])
        else:
            _run_observer(observer, acts)
            
def _pre_hook_func(config_node, module, inputs): # note inputs from module forward call, a tuple
    _FI_activations(config_node.activation, inputs)
    _FI_weights(config_node.weights, module)
    _run_observers(config_node.observers_pre, inputs)

def _after_hook_func(config_node, module, inputs, output):
    _FI_activations(config_node.activation_out, output)
    _run_observers(config_node.observers, output)

def _default_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

class ConfigItemList(list):
    """A convient list of detail config tree allow broadcasting assignment.
    
    example:
        ```python
        cfg = fi_model.get_configs('weights.0.selector')
        cfg.enable = False
        cfg[1].enable = True
        ```
    """
    def __getattr__(self, key):
        return ConfigItemList([x[key] for x in self])
    
    def __setattr__(self, key, value):
        if isinstance(value, list):
            assert len(value)==len(self), 'setattr value list should have same length as config item'
            for i, item in enumerate(self):
                item[key] = value[i]
            return

        for item in self:
            item[key] = value

def find_modules(model: Union['MRFI', torch.nn.Module], attrdict: dict):
    '''Find by `module_type`, `module_name`, `module_fullname`, also remove these attr in attrdict.'''
    moduletypes = _default_list(attrdict.pop('module_type', []))
    modulenames = _default_list(attrdict.pop('module_name', []))
    modulefullnames = _default_list(attrdict.pop('module_fullname', []))
    found = {}
    for name, module in model.named_modules():
        for typename in moduletypes:
            if typename in str(type(module)):
                found[name] = module
        for fname in modulenames:
            if fname in name:
                found[name] = module
        for fname in modulefullnames:
            if fname == name:
                found[name] = module
    if len(found)==0:
        logging.warning('mrfi.find_modules: No module match in current condition'
                        '(module_type: %s, module_name: %s, module_fullname: %s), please check.',
                        moduletypes, modulenames, modulefullnames)
    else:
        logging.info('mrfi.find_modules: Found modules %s.', list(found.keys()))
    return found

class MRFI:
    """MRFI core object, a wrapper of network module."""
    def __init__(self, model: nn.Module, config: FIConfig) -> None:
        self.model = model

        if isinstance(config, EasyConfig):
            self.config = self.__expand_config(config)
        elif isinstance(config, ConfigTree):
            self.config = config
        else:
            raise TypeError(config)

        self.__add_hooks()
        self.golden = False
        self.global_FI_enabled = True # also affect quantization

    def __add_hooks(self):
        for _, module in self.model.named_modules():
            module.register_forward_pre_hook(lambda mod, input: _pre_hook_func(mod.FI_config, mod, input))
            module.register_forward_hook(lambda mod, input, output: _after_hook_func(mod.FI_config, mod, input, output))

    def __empty_configtree(self, module: torch.nn.Module):
        curdict = {
            'weights': {},
            'activation_out':{},
            'activation':{},
            'observers':{},
            'observers_pre':{},
            'sub_modules':{}
        }
        for name, submodule in module.named_children():
            curdict['sub_modules'][name] = self.__empty_configtree(submodule)
        return curdict
    
    def __add_moduleconfig(self, configtree, module):
        module.FI_config = configtree
        for name, submodule in module.named_children():
            if name in configtree.sub_modules:
                self.__add_moduleconfig(configtree.sub_modules[name],submodule)
    
    def __change_arg(self, fiattr: dict):
        method = fiattr.pop('method')
        args = copy.copy(fiattr)
        fiattr.clear()
        fiattr['method'] = method
        fiattr['args'] = args
        
    def __config_fi(self, fi: dict, module, fitype):
        configtree = module.FI_config
        
        if fitype == 'activation' or fitype == 'activation_in':
            configtree.activation.append(
                ConfigTree(copy.deepcopy(fi), self, ConfigTreeNodeType.FI_ATTR, configtree.name + '.activation.0'))
        elif fitype == 'activation_out':
            configtree.activation_out.append(
                ConfigTree(copy.deepcopy(fi), self, ConfigTreeNodeType.FI_ATTR, configtree.name + '.activation_out.0'))
        elif fitype == 'weight':
            names = _default_list(fi.get('name', ['weight']))

            for i, name in enumerate(names):
                if not hasattr(module, name):
                    print('warning: no weight mode "{name}" in current module, ignore')
                    continue
                ficopy = copy.deepcopy(fi)
                ficopy['name'] = name
                configtree.weights.append(
                    ConfigTree(ficopy, self, ConfigTreeNodeType.FI_ATTR, configtree.name + '.weights.' + str(i)))
        else:
            raise ValueError('FI type shoude be either activation(_in)/actionvation_out/weight')
    
    @contextmanager
    def golden_run(self):
        """A context manager for golden run.
        
        Examples:
            >>> with fi_model.golden_run():
            >>>    out_golden = fi_model(inputs)
        """
        golden = self.golden
        self.golden = True
        yield
        self.golden = golden
    
    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwds):
        with torch.no_grad():
            return self.model(*args, **kwds)

    def get_configs(self, configname: str = None, strict: bool = False, **kwargs) -> ConfigItemList:
        """
        Find config tree nodes in current detail config.

        Args: 
            configname: A string splited by dot '.' to match ConfigTreeNode, e.g. 'activations.0.selector'.
            strict: Use strict match config name on found modules.
            
            module_name (Union[str, List[str]]): Found modules by module name, e.g. 'conv'.
            module_type (Union[str, List[str]]): Found modules by module type, e.g. 'Conv2d'.
            module_fullname (Union[str, List[str]]): Found modules by strict module name match, e.g. ['model.conv1', 'model.conv2'].
        """
        configname = configname.split('.')
        result = ConfigItemList()

        if len(kwargs) == 0:
            named_modules = dict(self.model.named_modules())
        else:
            named_modules = find_modules(self.model, kwargs)

        for module in named_modules.values():
            fi_config_root : ConfigTree = module.FI_config
            fi_config = fi_config_root
            for tname in configname:
                if fi_config.nodetype in (ConfigTreeNodeType.OBSERVER_LIST, ConfigTreeNodeType.FI_LIST):
                    try:
                        tname = int(tname)
                    except Exception:
                        raise ValueError('Config {fi_config} expect index, got {tname}') from None
                if not fi_config.hasattr(tname):
                    if strict:
                        raise ValueError(f'Config "{fi_config}" has no subnode named "{tname}"')
                    break
                fi_config = fi_config[tname]
            else:
                result.append(fi_config)
        return result
    
    def get_activation_configs(self, fi_configname: str = None, strict: bool = False, activation_id: int = 0, out: bool = False, **kwargs: dict):
        """A convenient function of `get_configs` to get activation config.

        Equivalent to call `get_configs('activation.0.{fi_configname}')` or `get_configs('activation_out.0.{fi_configname}').`
        """
        fi_configname = '' if fi_configname is None else '.' + fi_configname
        if out:
            return self.get_configs('activation_out.' + str(activation_id) + fi_configname, strict, **kwargs)
        else:
            return self.get_configs('activation.' + str(activation_id) + fi_configname, strict, **kwargs)
        
    def get_weights_configs(self, fi_configname: str = None, strict: bool = False, weight_id: int = 0, **kwargs: dict):
        """A convenient function of `get_configs` to get weights config.

        Equivalent to call `get_configs('weights.0.{fi_configname}')`
        """
        fi_configname = '' if fi_configname is None else '.' + fi_configname
        return self.get_configs('weights.' + str(weight_id) + fi_configname, strict, **kwargs)

    def save_config(self, filename: str):
        """Save current detail config tree to a file."""
        _write_config(self.config.state_dict(), filename)

    def __config_observer(self, observer:dict, module):
        configtree = module.FI_config

        observer_copy = copy.deepcopy(observer)
        pos = observer_copy.pop('position', 'after')

        if pos == 'pre':
            configtree.observers_pre.append(ConfigTree(observer_copy, self, ConfigTreeNodeType.OBSERVER_ATTR))
        elif pos == 'after':
            configtree.observers.append(ConfigTree(observer_copy, self, ConfigTreeNodeType.OBSERVER_ATTR))
        else:
            raise ValueError('observer position should be either pre/after')
        
    def __expand_config(self, config: EasyConfig):
        treedict = self.__empty_configtree(self.model)
        configtree = ConfigTree(treedict, self)
        self.__add_moduleconfig(configtree, self.model)

        filist:list[dict] = config.faultinject
        observerlist = config.observe

        for fi in filist:
            used_modules = find_modules(self.model, fi)
            
            fitype = fi.pop('type')
            activationid = fi.pop('activationid', 0)

            quantization = fi.get('quantization')
            if quantization is not None:
                self.__change_arg(quantization)
            selector = fi.get('selector')
            if selector is not None:
                self.__change_arg(selector)
            error_mode = fi.get('error_mode')
            if error_mode is not None:
                self.__change_arg(error_mode)

            for mod in used_modules.values():
                self.__config_fi(fi, mod, fitype)

        for observer in observerlist:
            used_modules = find_modules(self.model, observer)
            
            for mod in used_modules.values():
                self.__config_observer(observer, mod)

        return configtree

    def observers_reset(self) -> None:
        """Reset all observers in MRFI, usually executed before an epoch of observation."""
        for module in self.model.modules():
            config = module.FI_config
            if config.observers_pre is not None:
                for observer in config.observers_pre:
                    if observer.object is not None:
                        observer.object.reset()
            if config.observers is not None:
                for observer in config.observers:
                    if observer.object is not None:
                        observer.object.reset()

    def observers_result(self) -> dict:
        """A convient function to get all observer results in MRFI."""
        result = {}
        for module_name, module in self.model.named_modules():
            config = module.FI_config
            if config.observers_pre is not None:
                for i, observer in enumerate(config.observers_pre):
                    if observer.object is not None and hasattr(observer.object, 'result'):
                        name = module_name + '.pre.' + str(i)
                        res = observer.object.result()
                        result[name] = res

            if config.observers is not None:
                for i, observer in enumerate(config.observers):
                    if observer.object is not None and hasattr(observer.object, 'result'):
                        name = module_name + '.' + str(i)
                        res = observer.object.result()
                        result[name] = res
        
        return result

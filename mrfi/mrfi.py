import torch.nn as nn
import torch
import copy
from enum import Enum

class FIConfig:...

def config_loader(filename):
    extentname = filename.split('.')[-1]
    with open(filename) as f:
        if extentname == 'yml' or extentname=='yaml':
            import yaml
            config = yaml.full_load(f)
        else:
            raise NotImplementedError(extentname)
    return config

class EasyConfig(FIConfig):
    def __init__(self, config: dict):
        self.filist = config.get('faultinject', [])
        self.observerlist = config.get('observer', [])
    
    @classmethod
    def load_file(cls, filename):
        return cls(config_loader(filename))

class ConfigEnv:
    def __init__(self) -> None:
        self.golden = False

class ConfigTreeNodeType(Enum):
    MODULE = 0
    FI_LIST = 2
    OBSERVER_LIST = 3
    FI_ATTR = 4
    FI_STAGE = 5
    METHOD_ARGS = 6
    MODULE_LIST = 7
    OBSERVER_ATTR = 8

def node_step(nodetype, name):
    if nodetype == ConfigTreeNodeType.MODULE:
        if name == 'sub_modules':
            return ConfigTreeNodeType.MODULE_LIST
        if name in ('weights', 'activation', 'activation_out'):
            return ConfigTreeNodeType.FI_LIST
        if name in ('observers_pre', 'observers'):
            return ConfigTreeNodeType.OBSERVER_LIST
        raise ValueError(name)
    elif nodetype == ConfigTreeNodeType.FI_ATTR:
        if name in ('quantization', 'selector', 'modifier', 'enabled'):
            return ConfigTreeNodeType.FI_STAGE
        raise ValueError(name)
    elif nodetype == ConfigTreeNodeType.FI_STAGE:
        if name == 'method':
            return None
        if name == 'args':
            return ConfigTreeNodeType.METHOD_ARGS
        raise ValueError(name)
    elif nodetype == ConfigTreeNodeType.MODULE_LIST:
        return ConfigTreeNodeType.MODULE

class ConfigTree(FIConfig):
    def __init__(self, config_node, mrfi, nodetype = ConfigTreeNodeType.MODULE):
        self.__mrfi = mrfi
        self.__nodetype = nodetype
        if isinstance(config_node, list):
            subtype = {
                ConfigTreeNodeType.MODULE_LIST: ConfigTreeNodeType.MODULE,
                ConfigTreeNodeType.FI_LIST: ConfigTreeNodeType.FI_ATTR,
                ConfigTreeNodeType.OBSERVER_LIST: ConfigTreeNodeType.OBSERVER_ATTR,
            }[nodetype]
            self.raw_list = [ConfigTree(item, mrfi, subtype) for item in config_node]
            self.raw_dict = None

        elif isinstance(config_node, dict):
            self.raw_list = None
            self.raw_dict = {}
            for k, v in config_node.items():
                subtype = node_step(nodetype, k)
                if subtype != None:
                    self.raw_dict[k] = ConfigTree(v, mrfi, subtype)
                else:
                    self.raw_dict[k] = v
    
    def __contains__(self, key):
        if self.raw_list is None:
            return key in self.raw_dict
        return key in self.raw_list

    def __getitem__(self, id):
        if self.raw_list is None:
            return self.raw_dict[id]
        return self.raw_list[id]
    
    def __len__(self):
        if self.raw_list is None:
            return len(self.raw_dict)
        return len(self.raw_list)

    def __getattr__(self, name):
        if name == 'golden':
            return self.__mrfi.golden
        if self.raw_dict is None:
            raise AttributeError(name)
        if name in self.raw_dict:
            return self.raw_dict[name]
        raise KeyError(name)
    
    def get_dict(self):
        return self.raw_dict

    def get_list(self):
        return self.raw_list
    
    def state_dict(self):
        if self.raw_list is None:
            rdict = {}
            for k, v in self.raw_dict.items():
                if isinstance(v, ConfigTree):
                    rdict[k] = v.state_dict()
                else:
                    rdict[k] = v
            return rdict
        else:
            assert self.raw_list is not None
            return [x.state_dict() if isinstance(x, ConfigTree) else x for x in self.raw_list]


def FI_activation(config, act):
    if not config.enable: return
    
    layerwise_quantization, fi_quantization = False, False

    if config.quantization:
        if config.quantization.layerwise == False:
            fi_quantization = True
        else:
            layerwise_quantization = True
    
    if layerwise_quantization or fi_quantization:
        quantization_method = named_functions(config.quantization.method)
        quantization_args = copy.copy(config.quantization.args)
        if quantization_args.dynamic_range == 'max' or quantization_args.dynamic_range == 'auto':
            quantization_args.dynamic_range = act.abs().max()
    
    if layerwise_quantization:
        quantization_method.quantization(input, **quantization_args)
    
    if not config.golden:
        if config.selector is None:
            selector = named_functions(config.selector.method)
            selector_args = config.selector.args
            error_list = torch.as_tensor(selector(act, selector_args))
        else:
            error_list = slice(None)
        
        modifier = named_functions(config.modifier.method)
        modifier_args = config.modifier.args

        values = act.view(-1)[error_list]

        if fi_quantization: quantization_method.quantization(values)

        fi_value = modifier(values, **modifier_args)
        
        if fi_quantization: quantization_method.dequantization(fi_value)

        act.view(-1)[error_list] = fi_value

    if layerwise_quantization:
        quantization_method.dequantization(act, **quantization_args)

def FI_activations(config, acts):
    if isinstance(acts, tuple):
        assert((len(config)==1) or (len(config)==len(acts)), 
               'number of activation FI config should be 1 or same as module inputs')
        if len(config)==1:
            for act in acts:
                FI_activation(config[0], act)
        else:
            for i, act in enumerate(acts):
                FI_activation(config[i], act)
    else:
        FI_activation(config[0], acts)


def FI_weights(config, module):
    if config is None: return
    for weight_config in config:
        if weight_config.enabled == False: # not None, True
            continue
        weight = getattr(module, weight_config.name)

def run_observer(config, activation):
    if config.method is not None:
        observer_method = named_functions[config.method]
    observer_method(config.ctx, activation, config.golden)

def run_observers(config, act_value):
    for config_observer in config:
        run_observer(config_observer, act_value)

named_functions = {}

def add_function(name, function_object):
    named_functions[name] = function_object

def pre_hook_func(config_node, module, inputs): # note inputs from module forward call, a tuple
    FI_activations(config_node.activation, inputs)
    FI_weights(config_node.weights)
    run_observers(config_node.observers_pre, inputs)

def after_hook_func(config_node, module, inputs, output):
    FI_activations(config_node.activation_out, output)
    run_observers(config_node.observers_pre, output)

def default_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

class MRFI:
    def __init__(self, model: nn.Module, config: FIConfig) -> None:
        self.model = model

        if isinstance(config, EasyConfig):
            self.config = self.__expand_config(config)
        elif isinstance(config, ConfigTree):
            self.config = config
        else:
            raise TypeError(config)

        self.__add_hooks()

    def __add_hooks(self):
        for name, module in self.model.named_modules():
            module.register_forward_pre_hook(lambda mod, input: pre_hook_func(self, mod, input))
            module.register_forward_hook(lambda mod, input, output: after_hook_func(self, mod, input, output))
        
    @staticmethod
    def add_function(name, function_object):
        named_functions[name] = function_object

    def __findmodules(self, attrdict):
        '''find by name,type and remove item'''
        moduletypes = default_list(attrdict.pop('moduletype', []))
        modulenames = default_list(attrdict.pop('modulename', []))
        modulefullnames = default_list(attrdict.pop('modulefullname', []))
        found = {}
        for name, module in self.model.named_modules():
            for typename in moduletypes:
                if typename in str(type(module)):
                    found[name] = module
            for fname in modulenames:
                if fname in name:
                    found[name] = module
            for fname in modulefullnames:
                if fname == name:
                    found[name] = module
        return found

    def __empty_configtree(self, module: torch.nn.Module):
        curdict = {
            'weights': [],
            'activation_out':[],
            'activation':[],
            'observers':[],
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
        
    def __config_fi(self, fi:dict, module, fitype):
        configtree = module.FI_config
        
        print(fi)
        if fitype == 'activation' or fitype == 'activation_in':
            configtree.activation.raw_list.append(
                ConfigTree(copy.deepcopy(fi), self, ConfigTreeNodeType.FI_ATTR))
        elif fitype == 'activation_out':
            configtree.activation_out.raw_list.append(
                ConfigTree(copy.deepcopy(fi), self, ConfigTreeNodeType.FI_ATTR))
        elif fitype == 'weight':
            names = default_list(fi.pop('name', ['weight']))

            for name in names:
                if not hasattr(module, name):
                    print('warning: no "{name}" in current module, ignore')
                    continue
                ficopy = copy.deepcopy(fi)
                ficopy['name'] = name
                configtree.weight.raw_list.append(
                    ConfigTree(ficopy, self, ConfigTreeNodeType.FI_ATTR))

    def __config_observer(self, observer:dict, module):
        configtree = module.FI_config

        configtree.observers.raw_list.append(ConfigTree(observer, self, ConfigTreeNodeType.OBSERVER_ATTR))
        
    def __expand_config(self, config: EasyConfig):
        treedict = self.__empty_configtree(self.model)
        configtree = ConfigTree(treedict, self)
        print(configtree.state_dict())
        self.__add_moduleconfig(configtree, self.model)

        filist:list[dict] = config.filist
        observerlist = config.observerlist

        for fi in filist:
            used_modules = self.__findmodules(fi)
            
            fitype = fi.pop('type')
            activationid = fi.pop('activationid', 0)

            quantization = fi.get('quantization')
            if quantization is not None:
                self.__change_arg(quantization)
            selector = fi.get('selector')
            if selector is not None:
                self.__change_arg(selector)
            modifier = fi.get('modifier')
            if modifier is not None:
                self.__change_arg(modifier)

            for mod in used_modules.values():
                self.__config_fi(fi, mod, fitype)

        for observer in observerlist:
            used_modules = self.__findmodules(fi)
            
            for mod in used_modules.values():
                self.__config_observer(observer, mod)

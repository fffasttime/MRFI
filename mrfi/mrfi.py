import torch.nn as nn
import torch
import copy
from importlib import import_module
from enum import Enum
from contextlib import contextmanager

named_functions = {}

mrfi_packages = ['.observer', '.quantization', '.selector', '.error_mode']

def load_package_functions(name):
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

def add_function(name, function_object):
    named_functions[name] = function_object

class FIConfig:...

def read_config(filename):
    extentname = filename.split('.')[-1]
    with open(filename) as f:
        if extentname == 'yml' or extentname=='yaml':
            import yaml
            config = yaml.full_load(f)
        else:
            raise NotImplementedError(extentname)
    return config

def write_config(config: dict, filename):
    extentname = filename.split('.')[-1]
    with open(filename, 'w') as f:
        if extentname == 'yml' or extentname=='yaml':
            import yaml
            config = yaml.dump(config, f)
        else:
            raise NotImplementedError(extentname)
    return config

class EasyConfig(FIConfig):
    def __init__(self, config: dict):
        self.filist = config.get('faultinject', [])
        self.observerlist = config.get('observe', [])
    
    @classmethod
    def load_file(cls, filename):
        return cls(read_config(filename))

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
        if name in ('quantization', 'selector', 'error_mode'):
            return ConfigTreeNodeType.FI_STAGE
        if name in ('enabled', 'name'):
            return None
        raise ValueError(name)
    elif nodetype == ConfigTreeNodeType.FI_STAGE:
        if name == 'method':
            return None
        if name == 'args':
            return ConfigTreeNodeType.METHOD_ARGS
        raise ValueError(name)
    elif nodetype == ConfigTreeNodeType.MODULE_LIST:
        return ConfigTreeNodeType.MODULE
    elif nodetype == ConfigTreeNodeType.FI_LIST:
        return ConfigTreeNodeType.FI_ATTR
    elif nodetype == ConfigTreeNodeType.OBSERVER_LIST:
        return ConfigTreeNodeType.OBSERVER_ATTR

class ConfigTree(FIConfig):
    def __init__(self, config_node, mrfi, nodetype = ConfigTreeNodeType.MODULE, name = 'model'):
        self.__dict__['mrfi'] = mrfi
        self.__dict__['nodetype'] = nodetype
        self.__dict__['name'] = name
        if isinstance(config_node, list):
            raise ValueError(f"In config {self.name}, expect dict, got list")
        elif isinstance(config_node, dict):
            self.__dict__['raw_dict'] = {}
            for k, v in config_node.items():
                subtype = node_step(nodetype, k)
                if subtype != None:
                    self.raw_dict[k] = ConfigTree(v, mrfi, subtype, self.name + '.' + str(k))
                else:
                    self.raw_dict[k] = v
        
        self.__dict__['weight_copy'] = None
        self.__dict__['object'] = None
    
    def __contains__(self, key):
        return key in self.raw_dict

    def __getitem__(self, id):
        return self.raw_dict[id]
    
    def __setitem__(self, id, value):
        self.raw_dict[id] = value
    
    def __len__(self):
        return len(self.raw_dict)

    def __getattr__(self, name):
        if name == 'golden':
            return self.mrfi.golden
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
        return ''.join("<%s node '%s'>"%(self.nodetype.name, self.name))

    def append(self, obj):
        self.raw_dict[len(self.raw_dict)] = obj
    
    def state_dict(self):
        if self.nodetype == ConfigTreeNodeType.METHOD_ARGS:
            return self.raw_dict
        rdict = {}
        for k, v in self.raw_dict.items():
            if isinstance(v, ConfigTree):
                rdict[k] = v.state_dict()
            else:
                rdict[k] = v
        return rdict

    def hasattr(self, name):
        assert self.raw_dict is not None
        return name in self.raw_dict
    
    def getattr(self, name):
        if name not in self.raw_dict: return None
        return self.raw_dict[name]
    
    def set_internal_attr(self, key, name):
        self.__dict__[key] = name

def FI_activation(config, act):
    if not config.enabled: return
    
    layerwise_quantization, fi_quantization = False, False
    
    if config.quantization:
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
        quantization_method.quantization(act, **quantization_args)
    
    if not config.golden:
        if config.hasattr('selector'):
            selector = named_functions[config.selector.method]
            selector_args = config.selector.args.raw_dict
            error_list = torch.as_tensor(selector(act.shape, **selector_args))
        else:
            error_list = slice(None)
        
        modifier = named_functions[config.error_mode.method]
        modifier_args = config.error_mode.args.raw_dict

        values = act.view(-1)[error_list]

        if fi_quantization: quantization_method.quantization(values)

        fi_value = modifier(values, **modifier_args)
        
        if fi_quantization: quantization_method.dequantization(fi_value)

        act.view(-1)[error_list] = fi_value

    if layerwise_quantization:
        quantization_method.dequantization(act, **quantization_args)

def FI_activations(config, acts):
    if len(config)==0: return
    if isinstance(acts, tuple):
        assert (len(config)==1) or (len(config)==len(acts)), \
               'number of activation FI config should be 1 or same as module inputs'
        if len(config)==1:
            for act in acts:
                FI_activation(config[0], act)
        else:
            for i, act in enumerate(acts):
                FI_activation(config[i], act)
    else:
        FI_activation(config[0], acts)

def FI_weight(config, weight):
    if not config.enabled: return
    
    layerwise_quantization, fi_quantization = False, False
    
    if config.quantization:
        if config.quantization.hasattr('layerwise') and config.quantization.layerwise == False:
            fi_quantization = True
        else:
            layerwise_quantization = True
    
    if config.weight_copy == None:
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
        quantization_method.quantization(weight, **quantization_args)
    
    if not config.golden:
        if config.hasattr('selector'):
            selector = named_functions[config.selector.method]
            selector_args = config.selector.args.raw_dict
            error_list = torch.as_tensor(selector(weight.shape, **selector_args))
        else:
            error_list = slice(None)
        
        modifier = named_functions[config.error_mode.method]
        modifier_args = config.error_mode.args.raw_dict

        values = weight.view(-1)[error_list]

        if fi_quantization: quantization_method.quantization(values)

        fi_value = modifier(values, **modifier_args)
        
        if fi_quantization: quantization_method.dequantization(fi_value)

        weight.view(-1)[error_list] = fi_value

    if layerwise_quantization:
        quantization_method.dequantization(weight, **quantization_args)


def FI_weights(config, module):
    if config is None: return
    for weight_config in config:
        if weight_config.enabled == False: # not None, True
            continue
        weight = getattr(module, weight_config.getattr('name'))
        FI_weight(weight_config, weight)

def run_observer(config, activation):
    if config.object is None:
        observer_method = named_functions[config.method]
        config.set_internal_attr('object', observer_method())
    config.object.update(activation, config.golden)

def run_observers(config, acts):
    for observer in config:
        if isinstance(acts, tuple):
            index = int(observer.index) if observer.hasattr('index') else 0
            run_observer(observer, acts[index])
        else:
            run_observer(observer, acts)
            
def pre_hook_func(config_node, module, inputs): # note inputs from module forward call, a tuple
    FI_activations(config_node.activation, inputs)
    FI_weights(config_node.weights, module)
    run_observers(config_node.observers_pre, inputs)

def after_hook_func(config_node, module, inputs, output):
    FI_activations(config_node.activation_out, output)
    run_observers(config_node.observers, output)

def default_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

class ConfigItemList(list):
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
        self.golden = False

    def __add_hooks(self):
        for name, module in self.model.named_modules():
            module.register_forward_pre_hook(lambda mod, input: pre_hook_func(mod.FI_config, mod, input))
            module.register_forward_hook(lambda mod, input, output: after_hook_func(mod.FI_config, mod, input, output))
        
    @staticmethod
    def add_function(name, function_object):
        named_functions[name] = function_object

    def __findmodules(self, attrdict):
        '''find by name,type and remove item'''
        moduletypes = default_list(attrdict.pop('module_type', []))
        modulenames = default_list(attrdict.pop('module_name', []))
        modulefullnames = default_list(attrdict.pop('module_fullname', []))
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
        
    def __config_fi(self, fi:dict, module, fitype):
        configtree = module.FI_config
        
        if fitype == 'activation' or fitype == 'activation_in':
            configtree.activation.append(
                ConfigTree(copy.deepcopy(fi), self, ConfigTreeNodeType.FI_ATTR, configtree.name + '.activation.0'))
        elif fitype == 'activation_out':
            configtree.activation_out.append(
                ConfigTree(copy.deepcopy(fi), self, ConfigTreeNodeType.FI_ATTR, configtree.name + '.activation_out.0'))
        elif fitype == 'weight':
            names = default_list(fi.get('name', ['weight']))

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
        golden = self.golden
        self.golden = True
        yield
        self.golden = golden
    
    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwds):
        with torch.no_grad():
            return self.model(*args, **kwds)

    def get_configs(self, modulename = '', configname = None, strict = True) -> ConfigItemList:
        configname = configname.split('.')
        result = ConfigItemList()
        for name, module in self.named_modules():
            if modulename not in name: continue
            fi_config_root : ConfigTree = module.FI_config
            fi_config = fi_config_root
            for tname in configname:
                if fi_config.nodetype == ConfigTreeNodeType.OBSERVER_LIST or fi_config.nodetype == ConfigTreeNodeType.FI_LIST:
                    try:
                        tname = int(tname)
                    except:
                        raise ValueError('Config "%s" expect index, got "%s"'%(fi_config, tname))
                if not fi_config.hasattr(tname):
                    if strict:
                        raise ValueError('Config "%s" has no subnode named "%s"'%(fi_config, tname))
                    else:
                        break
                fi_config = fi_config[tname]
            else:
                result.append(fi_config)
        return result

    def save_config(self, filename):
        write_config(self.config.state_dict(), filename)

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
            error_mode = fi.get('error_mode')
            if error_mode is not None:
                self.__change_arg(error_mode)

            for mod in used_modules.values():
                self.__config_fi(fi, mod, fitype)

        for observer in observerlist:
            used_modules = self.__findmodules(observer)
            
            for mod in used_modules.values():
                self.__config_observer(observer, mod)

        return configtree

    def observers_reset(self) -> None:
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

    def observers_result(self) -> list:
        resultlist = [] # name, value
        for module_name, module in self.model.named_modules():
            config = module.FI_config
            if config.observers_pre is not None:
                for i, observer in enumerate(config.observers_pre):
                    if observer.object is not None and hasattr(observer.object, 'result'):
                        name = module_name + '.pre.' + str(i)
                        res = observer.object.result()
                        resultlist.append((name, res))

            if config.observers is not None:
                for i, observer in enumerate(config.observers):
                    if observer.object is not None and hasattr(observer.object, 'result'):
                        name = module_name + '.' + str(i)
                        res = observer.object.result()
                        resultlist.append((name, res))
        
        return resultlist

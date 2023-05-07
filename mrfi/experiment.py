
from .mrfi import MRFI, ConfigTree, ConfigItemList, ConfigTreeNodeType, named_functions, find_modules
import numpy as np
from typing import Optional, Union, List
import logging
import torch
import torch.utils.data

def logspace_density(low = -8, high = -4, density = 3):
    return np.logspace(low, high, density*(high - low) + 1)

def Acc_experiment(fi_model, dataloader):
    acc_fi, n_inputs = 0, 0
    device = next(fi_model.parameters()).device
    fi_model.observers_reset()

    for inputs, labels in dataloader:
        with torch.no_grad():
            outs = fi_model(inputs.to(device)).cpu()
        acc_fi += (outs.argmax(1)==labels).sum().item()
        n_inputs += len(labels)

    logging.info(f'mrfi.Acc_experinemt got {acc_fi / n_inputs}({acc_fi}/{n_inputs})')
    return acc_fi / n_inputs

def observeFI_experiment(fi_model: MRFI, 
                         input_data: Union[torch.Tensor, torch.utils.data.DataLoader],
                         use_golden: bool = True,
                         observers_dict: Optional[dict] = None):
    device = next(fi_model.parameters()).device
    fi_model.observers_reset()

    if isinstance(input_data, torch.Tensor):    
        with torch.no_grad():
            if use_golden:
                with fi_model.golden_run():
                    fi_model(input_data.to(device)).cpu()
            fi_model(input_data.to(device)).cpu()
    else:
        for inputs, labels in input_data:
            with torch.no_grad():
                if use_golden:
                    with fi_model.golden_run():
                        fi_model(inputs.to(device)).cpu()
                fi_model(inputs.to(device)).cpu()
    
    if observers_dict is None:
        result = fi_model.observers_result()
    else:
        result = {}
        for name, observer in observers_dict.items():
            result[name] = observer.object.result()

    logging.info(f'mrfi.ObserveFI_experiment got {result})')
    return result

def BER_Acc_experiment(fi_model: MRFI, 
                   fi_selectors, 
                   dataloader: torch.utils.data.DataLoader, 
                   BER_range = logspace_density(), 
                   bit_width = 16):
    BER_range = np.array(BER_range)
    if isinstance(fi_selectors, ConfigTree):
        fi_selectors = [fi_selectors]
    assert isinstance(fi_selectors, list), 'fi_selector should be either ConfigTree or ConfigItemList'
    assert fi_selectors[0].nodetype == ConfigTreeNodeType.FI_STAGE, 'should be a FI selector config node'
    assert fi_selectors[0].method == 'RandomPositionByRate', "selector in BER experiment should be 'RandomPositionByRate'"
    autoskip_larger = True

    Acc_result = np.zeros_like(BER_range)
    
    logging.info(f'run BER_Acc_experiment(BER_range={BER_range}, bit_width={bit_width})')

    for i, BER in enumerate(BER_range):

        fi_selectors.args.rate = bit_width * BER
        
        Acc_result[i] = Acc_experiment(fi_model, dataloader)

        if autoskip_larger and  Acc_result[i] < 0.01:
            logging.info('skip larger BER experiment')
            break
    
    return BER_range, Acc_result

def BER_observe_experiment(fi_model: MRFI, 
                   fi_selectors, 
                   dataloader: torch.utils.data.DataLoader, 
                   BER_range = logspace_density(), 
                   bit_width = 16,
                   use_golden = True):
    BER_range = np.array(BER_range)
    if isinstance(fi_selectors, ConfigTree):
        fi_selectors = [fi_selectors]
    assert isinstance(fi_selectors, list), 'fi_selector should be either ConfigTree or ConfigItemList'
    assert fi_selectors[0].nodetype == ConfigTreeNodeType.FI_STAGE, 'should be a FI selector config node'
    assert fi_selectors[0].method == 'RandomPositionByRate', "selector in BER experiment should be 'RandomPositionByRate'"

    observe_result = []
    
    logging.info(f'run BER_Acc_experiment(BER_range={BER_range}, bit_width={bit_width})')

    for i, BER in enumerate(BER_range):

        fi_selectors.args.rate = bit_width * BER
        
        observe_result.append(observeFI_experiment(fi_model, dataloader, use_golden))
    
    return BER_range, observe_result

def Acc_golden(fi_model: MRFI, dataloader: torch.utils.data.DataLoader, disable_quantization: bool = False):
    if disable_quantization:
        fi_model.global_FI_enabled = False

    with fi_model.golden_run():
        result = Acc_experiment(fi_model, dataloader)

    if disable_quantization:
        fi_model.global_FI_enabled = True
    
    return result

def temp_observer_expriment(fi_model, input_data, method, pre_hook, use_fi, **kwargs):
    if pre_hook:
        observer_hooks = fi_model.get_configs('observers_pre', **kwargs)
    else:
        observer_hooks = fi_model.get_configs('observers', **kwargs)

    ids = []
    obs_cfg = {}
    for observer_hook in observer_hooks:
        observer_node = ConfigTree({'method': method}, fi_model, ConfigTreeNodeType.OBSERVER_ATTR)
        ids.append(observer_hook.append(observer_node))
        name = observer_hook.name.replace('.sub_modules','').replace('.observers', '') + '.' + method
        obs_cfg[name] = observer_node
    
    result = observeFI_experiment(fi_model, input_data, use_fi, obs_cfg)
    
    for i, observer_hook in enumerate(observer_hooks):
        observer_hook.remove(ids[i])

    return result

def observeFI_experiment_plus(fi_model: MRFI, 
                        input_data: Union[torch.Tensor, torch.utils.data.DataLoader], 
                        method: str = 'RMSE', 
                        pre_hook: bool = False, 
                        **kwargs):
    return temp_observer_expriment(fi_model, input_data, method, pre_hook, True, **kwargs)
    
def get_activation_info(fi_model: MRFI, 
                        input_data: Union[torch.Tensor, torch.utils.data.DataLoader], 
                        method: str = 'MinMax', 
                        pre_hook: bool = False, 
                        **kwargs):

    fi_model.global_FI_enabled = False
    result = temp_observer_expriment(fi_model, input_data, method, pre_hook, False, **kwargs)
    fi_model.global_FI_enabled = True
    
    return result

def get_weight_info(model: Union[MRFI, torch.nn.Module], 
                     method: str = 'MinMax',
                     weight_name: Union[str, List[str]] = 'weight',
                     **kwargs): # module_name, module_type, module_fullname
    if isinstance(weight_name, str):
        weight_name = [weight_name]
        
    if len(kwargs) == 0:
        found_modules = dict(model.named_modules())
    else:
        found_modules = find_modules(model, kwargs)

    obs_result = {}
    for mname, module in found_modules.items():
        for wname in weight_name:
            if hasattr(module, wname) and isinstance(getattr(module, wname), torch.Tensor):
                w = getattr(module, wname)
                observer = named_functions[method]()
                observer.update(w)

                obs_result[mname + '.' + wname] = observer.result()
    
    return obs_result

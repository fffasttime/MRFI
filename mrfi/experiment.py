
from .mrfi import MRFI, ConfigTree, ConfigItemList, ConfigTreeNodeType, named_functions, find_modules
import numpy as np
from typing import Union, List
import logging
import torch

def logspace_density(low = -8, high = -4, density = 3):
    return np.logspace(low, high, density*(high - low) + 1)

def BER_Acc_experiment(fi_model: MRFI, fi_selectors, dataloader, BER_range = logspace_density(), bit_width = 16):
    BER_range = np.array(BER_range)
    if isinstance(fi_selectors, ConfigTree):
        fi_selectors = [fi_selectors]
    assert isinstance(fi_selectors, list), 'fi_selector should be either ConfigTree or ConfigItemList'
    assert fi_selectors[0].nodetype == ConfigTreeNodeType.FI_STAGE, 'should be a FI selector config node'
    assert fi_selectors[0].method == 'RandomPositionByRate', "selector in BER experiment should be 'RandomPositionByRate'"
    autoskip_larger = True

    Acc_result = np.zeros_like(BER_range)
    
    device = next(fi_model.parameters()).device
    logging.info(f'run BER_Acc_experiment(BER_range={BER_range}, bit_width={bit_width})')

    for i, BER in enumerate(BER_range):

        fi_selectors.args.rate = bit_width * BER

        fi_model.observers_reset()
        acc_fi, n_inputs = 0, 0
        for inputs, labels in dataloader:
            outs_fi = fi_model(inputs.to(device)).cpu()
            acc_fi += (outs_fi.argmax(1)==labels).sum().item()
            n_inputs += len(labels)
        
        Acc_result[i] = acc_fi / n_inputs
        logging.info(f'BER_Acc_experiment @BER = {BER}, Acc = {Acc_result[i]}({acc_fi}/{n_inputs})')

        if autoskip_larger and  Acc_result[i] < 0.01:
            logging.info('skip larger BER experiment')
            break
    
    return BER_range, Acc_result

def Acc_golden(fi_model: MRFI, dataloader, disable_quantization: bool = False):
    device = next(fi_model.parameters()).device

    if disable_quantization:
        fi_model.global_FI_enabled = False

    acc_fi, n_inputs = 0, 0

    for inputs, labels in dataloader:
        with fi_model.golden_run():
            with torch.no_grad():
                outs = fi_model(inputs.to(device)).cpu()
        acc_fi += (outs.argmax(1)==labels).sum().item()
        n_inputs += len(labels)
    
    if disable_quantization:
        fi_model.global_FI_enabled = True

    logging.info(f'Acc_golden got {acc_fi / n_inputs}({acc_fi}/{n_inputs})')
    return acc_fi / n_inputs

def benchmark_range(fi_model: MRFI, dataloader, module_name: Union[str, List[str]] = 'conv', pre_hook: bool = False, method: str = 'MinMax'):
    if pre_hook:
        observer_hooks = fi_model.get_configs(module_name, 'observers_pre', False)
    else:
        observer_hooks = fi_model.get_configs(module_name, 'observers', False)
    assert len(observer_hooks), 'No module found with name %s'%str(module_name)

    ids = []
    for observer_hook in observer_hooks:
        ids.append(observer_hook.append(ConfigTree({'method': method}, fi_model, ConfigTreeNodeType.OBSERVER_ATTR)))

    fi_model.global_FI_enabled = False
    fi_model.observers_reset()

    device = next(fi_model.parameters()).device

    logging.info(f'run benchmark_range, method={method}, observer_hooks={observer_hooks}')
    for inputs, labels in dataloader:
        with torch.no_grad():
            fi_model(inputs.to(device))

    fi_model.global_FI_enabled = True

    obs_result = {}
    
    for i, observer_hook in enumerate(observer_hooks):
        obs_result[observer_hook.name.replace('.sub_modules','').replace('.observers', '')] = observer_hook[ids[i]].object.result()
        observer_hook.remove(ids[i])

    return obs_result

def get_weight_range(model: Union[MRFI, torch.nn.Module], 
                     weight_name: Union[str, List[str]] = 'weight', 
                     method: str = 'MinMax',
                     **kwargs): # module_name, module_type, module_fullname
    if isinstance(weight_name, str):
        weight_name = [weight_name]
        
    found_modules = find_modules(model, kwargs)

    obs_result = {}
    for mname, module in found_modules.items():
        for wname in weight_name:
            if hasattr(module, wname) and isinstance(getattr(module, wname), torch.Tensor):
                w = getattr(module, wname)
                observer = named_functions[method]()
                observer.update(w, False)

                obs_result[mname + '.' + wname] = observer.result()
    
    return obs_result


from .mrfi import MRFI, ConfigTree, ConfigItemList, ConfigTreeNodeType
import numpy as np
from typing import Union, List

def logspace_density(low = -8, high = -4, density = 5):
    return np.logspace(low, high, density*(high - low) + 1)

def BER_Acc_experiment(fi_model: MRFI, fi_selectors, dataloader, BER_range = logspace_density(), bit_width = 16):
    BER_range = np.array(BER_range)
    if isinstance(fi_selectors, ConfigTree):
        fi_selectors = [fi_selectors]
    assert isinstance(fi_selectors, list), 'fi_selector should be either ConfigTree or ConfigItemList'
    assert fi_selectors[0].nodetype == ConfigTreeNodeType.FI_STAGE, 'should be a FI selector config node'
    assert fi_selectors[0].method == 'RandomPositionByRate', "selector in BER experiment should be 'RandomPositionByRate'"

    Acc_result = np.zeros_like(BER_range)

    for i, BER in enumerate(BER_range):
        fi_model.observers_reset()
        acc_fi, n_inputs = 0, 0

        for fi_config in fi_selectors:
            fi_selectors.args.rate = bit_width * BER

        for inputs, labels in dataloader:
            
            outs_fi = fi_model(inputs)
            acc_fi += (outs_fi.argmax(1)==labels).sum().item()
            n_inputs += len(labels)
        
        Acc_result[i] = acc_fi / n_inputs
    
    return BER_range * bit_width, Acc_result

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
    for inputs, labels in dataloader:
        fi_model(inputs)

    fi_model.global_FI_enabled = True

    obs_result = {}
    
    for i, observer_hook in enumerate(observer_hooks):
        obs_result[observer_hook.name.replace('.sub_modules','').replace('.observers', '')] = observer_hook[ids[i]].object.result()
        observer_hook.remove(ids[i])

    return obs_result

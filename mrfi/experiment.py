
from .mrfi import MRFI, ConfigTree, ConfigItemList, ConfigTreeNodeType, named_functions, find_modules
import numpy as np
from typing import Optional, Union, List
import logging
import torch
from torch.utils.data import DataLoader

def logspace_density(low: int = -8, high: int = -4, density: int = 3) -> np.array:
    """Similar as np.logspace but the datapoint always contains $10^{-n}$

    Args:
        low: begin point at `10**low`
        high: end point at `10**high`
        density: how many data point per 10x range.

    Returns:
        Array of uniformed logspace data point.
    """
    return np.logspace(low, high, density*(high - low) + 1)

def Acc_experiment(model: Union[MRFI, torch.nn.Module], dataloader: DataLoader) -> float:
    """Return classification accuracy on dataset.

    Args:
        model: Target model.
        dataloader: Yields a series of tuple of (batched) input images and classification label.

    Returns:
        Accuracy result among all data, between 0 and 1.
    """
    acc_fi, n_inputs = 0, 0
    device = next(model.parameters()).device
    if isinstance(model, MRFI):
        model.observers_reset()

    for inputs, labels in dataloader:
        with torch.no_grad():
            outs = model(inputs.to(device)).cpu()
        acc_fi += (outs.argmax(1)==labels).sum().item()
        n_inputs += len(labels)

    logging.info(f'mrfi.Acc_experinemt got {acc_fi / n_inputs}({acc_fi}/{n_inputs})')
    return acc_fi / n_inputs

def observeFI_experiment(fi_model: MRFI, 
                         input_data: Union[torch.Tensor, DataLoader],
                         use_golden: bool = True,
                         observers_dict: Optional[dict] = None) -> dict:
    """Run fault inject experiment and return internal oberver results.

    Args:
        fi_model: Target model.
        input_data: A input tensor or a dataloader.
        use_golden: If `True`, run both golden run and fault injection per batch,
            most of fault injection observer requires golden run before fault inject.
        observers_dict: A dict of observer config handlers to get result. 
            if `None`, return all observer result.

    Returns:
        Dictionary of observer result, keys are observer name and values are result of `observer.result()`.
    """
    device = next(fi_model.parameters()).device
    fi_model.observers_reset()

    if isinstance(input_data, torch.Tensor):    
        with torch.no_grad():
            if use_golden:
                with fi_model.golden_run():
                    fi_model(input_data.to(device))
            fi_model(input_data.to(device))
    else:
        for inputs in input_data:
            if isinstance(inputs, (List, tuple)): inputs = inputs[0] # ignore label
            with torch.no_grad():
                if use_golden:
                    with fi_model.golden_run():
                        fi_model(inputs.to(device))
                fi_model(inputs.to(device))
    
    if observers_dict is None:
        result = fi_model.observers_result()
    else:
        result = {}
        for name, observer in observers_dict.items():
            result[name] = observer.object.result()

    logging.info(f'mrfi.ObserveFI_experiment got {result})')
    return result

def BER_Acc_experiment(fi_model: MRFI, 
                   fi_selectors: ConfigItemList, 
                   dataloader: DataLoader, 
                   BER_range: Union[list, np.array] = logspace_density(), 
                   bit_width: int = 16) -> tuple:
    """Conduct a series of experimet under different bit error rate and get accuracy.

    Args:
        fi_model: Target model.
        fi_selectors: A target selector list from `fi_model.get_configs()`.
        dataloader: Yields a series of tuple of (batched) input images and classification label.
        BER_range: A list or a array for target bit error rate.
        bit_width: To calculate bit error rate by tensor value selected rate.

    Returns:
        Current bit error rate list and accuracy result among all data of each bit error list.
    """
    BER_range = np.array(BER_range)
    if isinstance(fi_selectors, ConfigTree):
        fi_selectors = [fi_selectors]
    if not isinstance(fi_selectors, list):
        raise RuntimeError('fi_selector should be either ConfigTree or ConfigItemList')
    if fi_selectors[0].nodetype != ConfigTreeNodeType.FI_STAGE:
        raise RuntimeError('fi_selector should be a FI selector config node')
    if 'Rate' not in fi_selectors[0].method:
        raise RuntimeError(f"Selector in BER experiment should be a rate selector, but get {fi_selectors[0].method}.")
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
                   fi_selectors: ConfigItemList, 
                   dataloader: DataLoader, 
                   BER_range: Union[list, np.array] = logspace_density(), 
                   bit_width: int = 16,
                   use_golden: bool = True) -> tuple:
    """Conduct a series of experimet under different bit error rate and get all observers result.

    Args:
        fi_model: Target model.
        fi_selectors: A target selector list from `fi_model.get_configs()`.
        dataloader: Yields a series of tuple of (batched) input images and classification label.
        BER_range: A list or a array for target bit error rate.
        bit_width: To calculate bit error rate by tensor value selected rate.
        use_golden: If `True`, run both golden run and fault injection per batch.
        
    Returns:
        Current bit error rate list and observation result dictionary list.
    """
    BER_range = np.array(BER_range)
    if isinstance(fi_selectors, ConfigTree):
        fi_selectors = [fi_selectors]
    if not isinstance(fi_selectors, list):
        raise RuntimeError('fi_selector should be either ConfigTree or ConfigItemList')
    if fi_selectors[0].nodetype != ConfigTreeNodeType.FI_STAGE:
        raise RuntimeError('fi_selector should be a FI selector config node')
    if 'Rate' not in fi_selectors[0].method:
        raise RuntimeError(f"Selector in BER experiment should be a rate selector, but get {fi_selectors[0].method}.")

    observe_result = []
    
    logging.info(f'run BER_Acc_experiment(BER_range={BER_range}, bit_width={bit_width})')

    for BER in BER_range:

        fi_selectors.args.rate = bit_width * BER
        
        observe_result.append(observeFI_experiment(fi_model, dataloader, use_golden))
    
    return BER_range, observe_result

def Acc_golden(fi_model: MRFI, dataloader: torch.utils.data.DataLoader, disable_quantization: bool = False) -> float:
    """Evaluate model accuracy without error injection.

    Args:
        fi_model: Target model.
        dataloader: Yields a series of tuple of (batched) input images and classification label.
        disable_quantization: If `True`, also disable quantization if it exists. 
                              Note that quantization also affects model accuracy.
    
    Returns:
        Golden run accuracy result among all data, between 0 and 1.         
    """
    if disable_quantization:
        fi_model.global_FI_enabled = False

    with fi_model.golden_run():
        result = Acc_experiment(fi_model, dataloader)

    if disable_quantization:
        fi_model.global_FI_enabled = True
    
    return result

def _temp_observer_expriment(fi_model, input_data, method, pre_hook, use_fi, **kwargs):
    """Add temporary observer to model, conduct experiment to get result, them remove them."""
    if not kwargs: kwargs = {'module_name':''}

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
                              input_data: Union[torch.Tensor, DataLoader], 
                              method: str = 'RMSE', 
                              pre_hook: bool = False, 
                              **kwargs: dict) -> dict:
    """Run fault injection experiment and observe its internal effect.

    Compare with `observeFI_experiment()`, this function add temporary target observers of to model.

    Args:
        fi_model: Target model.
        input_data: A input tensor or a dataloader.
        method: Name of the fault injection observer in `mrfi.observer`.
        pre_hook: Observer value in module input, not output.
        **kwargs: Target module to do observe, `module_type`, `module_name` or `module_fullname`
        
    Returns:
        Dictionary of observer result, keys are observer name and values are result of `observer.result()`.
    """
    return _temp_observer_expriment(fi_model, input_data, method, pre_hook, True, **kwargs)

def get_activation_info(fi_model: MRFI, 
                        input_data: Union[torch.Tensor, DataLoader], 
                        method: str = 'MinMax', 
                        pre_hook: bool = False, 
                        **kwargs: dict) -> dict:
    """Observe model activations without fault injection.

    Args:
        fi_model: Target model.
        input_data: A input tensor or a dataloader.
        method: Name of the basic observer in `mrfi.observer`.
        pre_hook: Observer value in module input, not output.
        **kwargs: Target module to observe, `module_type`, `module_name` or `module_fullname`

    Returns:
        Dictionary of observer result, keys are activation observation name and values are result of `observer.result()`.
    """
    fi_model.global_FI_enabled = False
    result = _temp_observer_expriment(fi_model, input_data, method, pre_hook, False, **kwargs)
    fi_model.global_FI_enabled = True
    
    return result

def get_weight_info(model: Union[MRFI, torch.nn.Module], 
                     method: str = 'MinMax',
                     weight_name: Union[str, List[str]] = 'weight',
                     **kwargs: dict) -> dict:
    """Observe model weights.

    Args:
        model: Target model.
        method: Name of the basic observer in `mrfi.observer`.
        weight_name: Target weight name.
        **kwargs: Target module to observe, `module_type`, `module_name` or `module_fullname`

    Returns:
        Dictionary of observer result, keys are weight observation and values are result of `observer.result()`.
    """
    if isinstance(weight_name, str):
        weight_name = [weight_name]
        
    if not kwargs:
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

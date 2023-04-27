
from .mrfi import MRFI, ConfigTree, ConfigItemList, ConfigTreeNodeType
import numpy as np

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


from dataset.imagenet import make_testloader
from mrfi import MRFI, EasyConfig
from mrfi.experiment import get_activation_info, observeFI_experiment_plus, Acc_experiment
from torchvision.models import vgg11
import torch
import numpy as np

batch_size = 4
n_images = 10000

config = """
faultinject:
  - type: activation
    quantization:
      method: SymmericQuantization
      dynamic_range: auto
      bit_width: 16
    enabled: True
    selector:
      method: RandomPositionByNumber
      n: 1
    error_mode:
      method: IntSignBitFlip
      bit_width: 16

    module_fullname: features.0
"""

ACC = True

if ACC:
    # fi_model = MRFI(vgg11(pretrained=True).cuda().eval(), EasyConfig())
    # print(Acc_experiment(fi_model, make_testloader(n_images, batch_size=128)))

    econfig = EasyConfig.load_string(config)
    fi_model = MRFI(vgg11(pretrained=True).cuda().eval(), econfig)
    print(Acc_experiment(fi_model, make_testloader(n_images, batch_size=128)))
    
    econfig = EasyConfig.load_string(config)
    econfig.faultinject[0]['type'] = 'weight'
    fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), econfig)
    print(Acc_experiment(fi_model, make_testloader(n_images, batch_size=128)))

else:
    res = []
    fi_model = MRFI(vgg11(pretrained=True).eval(), EasyConfig())
    # print(get_activation_info(fi_model, make_testloader(128, batch_size = batch_size), module_type = ['Conv', 'Linear']))
    STD = np.array(list(get_activation_info(fi_model, make_testloader(n_images, batch_size = batch_size), 'Std', module_type = ['Conv', 'Linear']).values()))
    print(STD)
    MABS = np.array(list(get_activation_info(fi_model, make_testloader(n_images, batch_size = batch_size), 'MeanAbs', module_type = ['Conv', 'Linear']).values()))
    print(MABS)
    res.append(STD)
    res.append(MABS)

    shape = (get_activation_info(fi_model, make_testloader(1, batch_size = 1), 'Shape'))
    # for k, v in shape.items(): print(k, v, v.numel())

    econfig = EasyConfig.load_string(config)
    fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), econfig)
    RMSE = np.array(list(observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), 'RMSE', module_type = ['Conv', 'Linear']).values()))
    print('act RMSE', RMSE)
    res.append(RMSE)
    print(RMSE/STD)
    res.append(RMSE/STD)

    MAE = np.array(list(observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), 'MAE', module_type = ['Conv', 'Linear']).values()))
    print('act MAE', MAE)
    res.append(MAE)
    print(MAE/MABS)
    res.append(MAE/MABS)

    econfig = EasyConfig.load_string(config)
    econfig.faultinject[0]['type'] = 'weight'
    fi_model = MRFI(vgg11(pretrained = True).cuda().eval(), econfig)
    RMSE = np.array(list(observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), 'RMSE', module_type = ['Conv', 'Linear']).values()))
    print('w RMSE', RMSE)
    res.append(RMSE)
    print(RMSE/STD)
    res.append(RMSE/STD)

    MAE = np.array(list(observeFI_experiment_plus(fi_model, make_testloader(n_images, batch_size = batch_size), 'MAE', module_type = ['Conv', 'Linear']).values()))
    print('w MAE', MAE)
    res.append(MAE)
    print(MAE/MABS)
    res.append(MAE/MABS)

    np.savetxt('result/vgg11_spread.txt', np.array(res).T, header='STD, MABS, A RMSE, A RRMSE, A MAE, A RMAE, W RMSE, W RRMSE, W MAE, W RMAW')

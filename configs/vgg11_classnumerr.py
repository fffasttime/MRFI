import yaml
import logging
import torch
import numpy as np
from scipy.stats import hypergeom

import mrfi.observer
from model.vgg11 import Net, testset
from mrfi.injector import ModuleInjector

yamlcfg='''
FI_activation: true
FI_enable: true
FI_weight: false
flip_mode: flip_int_highest
flip_mode_args:
  bit_width: 16
layerwise_quantization:
  bit_width: 16
  dynamic_range: 64
selector: RandomPositionSelector_Rate
selector_args:
  rate: 0.0001
sub_modules:
  features:
    sub_modules:
      0:
        FI_enable: false
      3:
      6:
      8:
      11:
      13:
      16:
      18:
observer:
    map: mse
    reduce: sum
'''

def classprederr(out, label):
    # 平均正确率：在M=999个非label输出中有n个大于label的值，但随机选N个都没选到大于label的数的情况，N+1为实际需要的分类数量。
    # 即超几何分布中k=0
    n = np.count_nonzero(out>out[label])
    prob = hypergeom.pmf(0, 999, n, np.arange(1000))
    return prob

def exp(total = 10000):
    torch.set_num_threads(16)
    config = yaml.load(yamlcfg)

    net=Net(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    FI_network = ModuleInjector(net, config)

    rates=np.logspace(-6, -3, 31)


    for mode in [mrfi.flip_mode.flip_int_highest]:

        classnum_err = np.zeros((len(rates), 1000))

        for ri, rate in enumerate(rates):
            data=iter(testloader)

            for layer in FI_network.features.subinjectors:
                layer.selector_args['rate']=rate
                layer.selector_args['poisson']=True
                layer.update_selector()
                layer.flip_mode = mode

            FI_network.reset_observe_value()

            acc=0
            for i in range(total):
                images, labels = next(data)
                images=images.to(device)
                out_free=FI_network(images, golden=True).cpu().numpy()
                out=FI_network(images).cpu().numpy()
                acc+=(np.argmax(out[0])==labels.numpy()[0])

                classnum_err[ri, :] += classprederr(out[0], labels.item()) # sum

            classnum_err[ri]/=total # average acc
            
            observes=FI_network.get_observes()

            print('%.10f'%rate, end='\t')
            for name, value in observes.items():
                print("%.4f"%np.sqrt(value/total), end='\t')
            print('%.2f%%'%(acc/total*100), flush=True)

            np.save('vgg11_classnumerr.npy', classnum_err)

from dataset.lenet_cifar import make_testloader, LeNet
from mrfi import MRFI, EasyConfig
from mrfi.experiment import observeFI_experiment, observeFI_experiment_plus
import matplotlib.pyplot as plt
import torch
from pprint import pprint

easyconfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
easyconfig.faultinject[0]['selector'] = {'method':'RandomPositionByNumber', 'n':1}
easyconfig.observe.append({'method':'RMSE','module_type':['Conv','Linear']})

fi_model = MRFI(LeNet(trained=True), easyconfig)
input_images = make_testloader(1, batch_size = 128)

fi_configs = fi_model.get_configs('activation.0')
print(fi_configs)

fi_configs.enabled = False
fi_configs[0].enabled = True

pprint(observeFI_experiment(fi_model, input_images))

pprint(observeFI_experiment_plus(fi_model, input_images, method = 'MAE', module_type = ['Conv', 'Linear']))
pprint(observeFI_experiment_plus(fi_model, input_images, method = 'EqualRate', module_type = ['Conv', 'Linear']))
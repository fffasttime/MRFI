from dataset.lenet_cifar import make_testloader, LeNet
from mrfi import MRFI, EasyConfig
from mrfi.experiment import get_activation_info, get_weight_info
import matplotlib.pyplot as plt
import torch
from pprint import pprint

fi_model = MRFI(LeNet(trained=True), EasyConfig())
input_images = make_testloader(128, batch_size = 128)

print("Activation Shape:")
pprint(get_activation_info(fi_model, torch.zeros([1,3,32,32]), 'Shape'))
pprint("Weight Shape:")
pprint(get_weight_info(fi_model, 'Shape'))

print("Activation MinMax:")
pprint(get_activation_info(fi_model, input_images))
print("Weight MinMax:")
pprint(get_weight_info(fi_model))

print("Activation AbsMax:")
pprint(get_activation_info(fi_model, input_images, 'MaxAbs'))
print("Weight AbsMax:")
pprint(get_weight_info(fi_model, 'MaxAbs'))

print("Activation Std:")
pprint(get_activation_info(fi_model, input_images, 'Std'))
print("Weight Std:")
pprint(get_weight_info(fi_model, 'Std'))

plt.style.use('seaborn')

act_samping = get_activation_info(fi_model, input_images, 'UniformSampling', module_name = ['conv', 'fc'])
fig, axs = plt.subplots(2,3)
for i, (name, value) in enumerate(act_samping.items()):
    ax = axs[i//3, i%3]
    ax.hist(value, bins=20)
    ax.set_title(name)
axs[1,2].violinplot(list(act_samping.values()))
plt.show()

weight_samping = get_weight_info(fi_model, 'UniformSampling')
fig, axs = plt.subplots(2,3)
for i, (name, value) in enumerate(weight_samping.items()):
    ax = axs[i//3, i%3]
    ax.hist(value, bins=20)
    ax.set_title(name)
axs[1,2].violinplot(list(weight_samping.values()))
plt.show()

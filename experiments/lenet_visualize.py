from dataset.lenet_cifar import make_testloader, LeNet, testset
from mrfi import MRFI, EasyConfig
from mrfi.experiment import get_activation_info, observeFI_experiment_plus, Acc_golden, Acc_experiment
import matplotlib.pyplot as plt
import torch
from pprint import pprint
import numpy as np

easyconfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
# pos = np.random.randint(0, 3*32*32)
pos = 674

easyconfig.set_selector(0, {'method':'FixPosition', 'position': pos})
del easyconfig.faultinject[0]['module_type']
easyconfig.faultinject[0]['module_name'] = 'conv1'
# easyconfig.observe.append({'method':'RMSE','module_type':['Conv','Linear']})

fi_model = MRFI(LeNet(trained=True), easyconfig)

def find_diff():
    testloader = make_testloader(10000)

    for i, (inputs, labels) in enumerate(testloader):
        with fi_model.golden_run():
            out_g = fi_model(inputs)[0]
            cls_g = out_g.argmax().item()
        
        out = fi_model(inputs)[0]
        cls = out.argmax().item()

        lbl = labels[0].item()

        if cls_g != cls and cls_g == lbl:
            print(i, cls, cls_g, lbl)
            print(out, out_g)

img, lbl = testset[183]

print(img.shape, lbl)
img = img.reshape((1, *img.shape))

with fi_model.golden_run():
    out_g = fi_model(img)[0]
out = fi_model(img)[0]

print(out_g, out_g.argmax().item(), out_g.softmax(0))
print(out, out.argmax().item(), out.softmax(0))

print(get_activation_info(fi_model, img, 'Shape'))
print(observeFI_experiment_plus(fi_model, img))
print(observeFI_experiment_plus(fi_model, img, 'EqualRate'))


datas = list(observeFI_experiment_plus(fi_model, img, 'SaveLast', module_type = ['Linear','Conv']).values())

def visualize_feature_map(x: torch.Tensor):
    x = x.squeeze().numpy()
    if len(x.shape) == 3:
        C,H,W = x.shape
        d = int(np.ceil(np.sqrt(C)))

        pic = np.zeros((d*H, d*W))

        for i in range(C):
            row, col = i//d, i%d
            pic[row*H:row*H+H, col*W:col*W+W] = x[i]

    else:
        n = len(x)
        if n>10:
            d = int(np.ceil(np.sqrt(n)))
            pic = np.zeros(d*d)
            pic[:n] = x
            pic = pic.reshape(d, d)
            
        else:
            pic = np.zeros((1,n))
            pic[0] = x            

    im = plt.imshow(pic, cmap='gray')
    plt.colorbar(im)

titles = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']

for i, (g, e) in enumerate(datas):
    plt.subplot(2, 5, i+1)
    visualize_feature_map(g)
    plt.title(titles[i])

    plt.subplot(2, 5, i+6)
    visualize_feature_map(g - e)
    plt.title(titles[i] + ' err')

plt.show()
# pprint(observeFI_experiment(fi_model, input_images))

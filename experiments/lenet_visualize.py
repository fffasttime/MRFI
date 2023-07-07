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

def visualize_feature_map(x: torch.Tensor, square = False, cmap = 'gray', symmeric = False):
    def best_hw(n):
        d = int(np.floor(np.sqrt(n)))
        for dh in range(d, 0, -1):
            if n%dh == 0:
                break
        dw = n//dh
        if square or dh < d*0.5: 
            return d, d
        return dh, dw

    x = x.squeeze().numpy()
    if len(x.shape) == 3:
        C,H,W = x.shape
        dh, dw = best_hw(C)

        pic = np.zeros((dh*H, dw*W))

        for i in range(C):
            row, col = i//dw, i%dw
            pic[row*H:row*H+H, col*W:col*W+W] = x[i]
        
        from matplotlib import ticker
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(W))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(H))

    else:
        n = len(x)
        if n>5 or square:
            dh, dw = best_hw(n)
            pic = np.zeros(dh*dw)
            pic[:n] = x
            pic = pic.reshape(dh, dw)
        else:
            pic = np.zeros((1,n))
            pic[0] = x            

    vmin, vmax = np.min(x), np.max(x)
    if symmeric:
        absmax = np.max(np.abs(x))
        vmin, vmax = -absmax, absmax

    im = plt.imshow(pic, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar(im, fraction = 0.05, pad = 0.05)

titles = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
N = len(titles)

for i, (g, e) in enumerate(datas):
    plt.subplot(2, N, i+1)
    visualize_feature_map(g, cmap = 'gray')
    plt.title(titles[i] + ' activation')

    plt.subplot(2, N, i+1+N)
    visualize_feature_map(g - e, cmap = 'RdBu', symmeric = True)
    plt.title(titles[i] + ' error')
    if i == N - 1: break

plt.show()
# pprint(observeFI_experiment(fi_model, input_images))

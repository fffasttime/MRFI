import sys

sys.path.append('.')
from mrfi import experiment
import torch
from dataset.lenet_cifar import make_testloader, Net
from mrfi import MRFI, EasyConfig, load_package_functions, add_function

class NoQuantization:
    @staticmethod
    def quantize(x, dynamic_range, bit_width):
        ...

    @staticmethod
    def dequantize(x, dynamic_range, bit_width):
        ...

yamlstr = '''
faultinject:
  - type: activation
    quantization:
      method: NoQuantization
      layerwise: False
      dynamic_range: auto
      bit_width: 16
    module_name: [fc1, fc2]
    enabled: False
    error_mode: 
      method: TestErrorModel
observe:
  - method: MinMax
    module_name: [fc2]
    position: pre
'''

def test_01():
    load_package_functions('test.custom_function')
    add_function('NoQuantization', NoQuantization)
    econfig = EasyConfig.load_string(yamlstr)

    fi_model = MRFI(Net(trained=True), econfig)
    print(fi_model(torch.zeros(1,3,32,32)))

    cfg = fi_model.get_configs('activation.0')
    cfg.enabled = [True, True]
    fi_model.observers_reset()
    print(fi_model(torch.zeros(1,3,32,32)))

    print(fi_model.observers_result())

    fi_model.save_config('.pytest_cache/test_addfunc_output.yaml')

    fi_model = MRFI(Net(trained=True), '.pytest_cache/test_addfunc_output.yaml')


def test_02():
    econfig = EasyConfig.load_string(yamlstr)
    econfig.faultinject[0]['type'] = 'weight'
    econfig.set_module_used(0, module_fullname=['fc1','fc2'])

    fi_model = MRFI(Net(trained=True), econfig)

    cfg = fi_model.get_configs('weights.0')
    cfg.enabled = [False, True]
    fi_model.observers_reset()
    print(fi_model(torch.zeros(1,3,32,32)))

    print(fi_model.observers_result())

if __name__ =='__main__':
    import logging
    logging.basicConfig(level = logging.INFO)
    test_01()
    test_02()

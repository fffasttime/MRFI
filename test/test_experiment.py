import sys

sys.path.append('.')
from mrfi import experiment
import torch
from dataset.lenet_cifar import make_testloader, Net
from mrfi import MRFI, EasyConfig

def test_00():
    econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
    fi_model = MRFI(Net(trained=True), econfig)
    assert len(experiment.get_weight_info(fi_model, 'Shape')) == 5
    assert list(experiment.get_weight_info(fi_model, 'Shape', module_fullname = 'fc2').values())[0] == torch.Size((84,120))
    experiment.get_activation_info(fi_model, make_testloader(4, batch_size = 2), 'Shape', module_type = 'Linear')

def test_01():
    econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
    del econfig.faultinject[0]['quantization']
    econfig.set_selector(0, {'method':'FixPositions', 'positions':[[0, 1], [2, 3]]})
    econfig.set_error_mode(0, {'method':'SetValue', 'value':100})
    econfig.set_module_used(0, module_name='fc3')
    econfig.faultinject[0]['enabled'] = False
    econfig.faultinject[0]['type'] = 'activation_out'
    econfig.observe.append({'method':'MaxAbs', 'module_fullname':'fc3'})
    fi_model = MRFI(Net(trained=True), econfig)

    assert fi_model.fc3.FI_config.activation_out[0].enabled == False

    dataloader = make_testloader(4, batch_size = 2)

    for x, label in dataloader:
        print(fi_model(x), label)

    fi_model.fc3.FI_config.activation_out[0].enabled = True
    assert fi_model.fc3.FI_config.activation_out[0].enabled == True

    dataloader = make_testloader(4, batch_size = 2)
    fi_model.observers_reset()
    
    for x, label in dataloader:
        out = fi_model(x)
        assert out[0, 2] == 100
        assert out[1][3] == 100

    assert experiment.Acc_golden(fi_model, make_testloader(4, batch_size = 2), disable_quantization = True) == .5
    assert experiment.Acc_experiment(fi_model, make_testloader(4, batch_size = 2)) == 0

    assert list(fi_model.observers_result().values())[0] == 100
    assert list(experiment.observeFI_experiment(fi_model, make_testloader(4, batch_size = 2), False).values())[0] == 100

def test_02():
    econfig = EasyConfig.load_file('easyconfigs/weight_fi.yaml')
    fi_model = MRFI(Net(trained=True), econfig)

    print(experiment.BER_Acc_experiment(fi_model, 
                                        fi_model.get_weights_configs('selector'), 
                                        make_testloader(10, batch_size = 5), 
                                        experiment.logspace_density(-5, -2, 3)))

def test_03():
    econfig = EasyConfig.load_file('easyconfigs/fxp_fi.yaml')
    econfig.set_quantization(0, {'integer_bit':4}, True)
    econfig.observe.append({'method':'RMSE', 'module_name':'fc3'})

    fi_model = MRFI(Net(trained=True), econfig)
    experiment.BER_observe_experiment(fi_model, 
                                      fi_model.get_activation_configs('selector'), 
                                      make_testloader(10, batch_size = 5), 
                                      experiment.logspace_density(-5, -3, 3))
    
    fi_model.observers_reset()
    experiment.observeFI_experiment_plus(fi_model, torch.zeros(1,3,32,32), pre_hook=True, module_name = 'fc2')

if __name__ =='__main__':
    import logging
    logging.basicConfig(level = logging.INFO)
    test_01()
    test_02()
    test_03()

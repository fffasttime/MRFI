# Validate fault injection to yolov8 by one line config
# Following yolov8's official docs https://docs.ultralytics.com/modes/val/
from ultralytics import YOLO
from mrfi import MRFI, EasyConfig

# Load a model
model = YOLO('experiments/yolov8n.pt')  # load an official model
econfig = EasyConfig.load_file('easyconfigs/default_fi.yaml')
# econfig.faultinject[0]['error_mode']['method'] = 'IntRandomBitFlip'
# econfig.faultinject[0]['quantization']['scale_factor'] = 1
# econfig.faultinject[0]['selector']['rate'] = 1.6e-4
fi_model = MRFI(model.model, econfig) # YOLO.model is real pytorch model
# fi_model.save_config('detailconfigs/yolov8n_dt.yaml') # save detail config

# Validate the model
metrics = model.val(data='coco128.yaml') # 128 images
print('map50:', metrics.box.map50) # got 0.03~0.06

with fi_model.golden_run():
    metrics = model.val(data='coco128.yaml')
    print('map50:', metrics.box.map50) # got 0.60

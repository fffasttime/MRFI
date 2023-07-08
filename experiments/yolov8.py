# Validate fault injection to yolov8 by one line config
# Following yolov8's official docs https://docs.ultralytics.com/modes/val/
from ultralytics import YOLO
from mrfi import MRFI, EasyConfig

# Load a model
model = YOLO('experiments/yolov8n.pt')  # load an official model
fi_model = MRFI(model.model, EasyConfig.load_file('easyconfigs/default_fi.yaml')) # YOLO.model is real pytorch model
# fi_model.save_config('detailconfigs/yolov8n_dt.yaml') # save detail config

# Validate the model
metrics = model.val(data='coco128.yaml') # 128 images
print(metrics.box.map50) # got 0.03~0.06

with fi_model.golden_run():
    metrics = model.val(data='coco128.yaml')
    print(metrics.box.map50) # got 0.60

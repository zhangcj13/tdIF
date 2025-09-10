# tdIF
The source code for paper: Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection

To view our paper, please refer: [Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection](https://arxiv.org/abs/2508.20392). 

## Method
![image](https://github.com/zhangcj13/tdIF/blob/main/images/overview.png)

## Prepare Python env 
python=3.8.13
```
# install torch
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# install yolox
bash llibs/download_install.sh
# install packages
pip install -r requirement.txt
```
## Prepare datasets
### Expected dataset structure for [COCO detection](https://cocodataset.org/#download):
```
COCO/
  annotations/instances_{train,val}2017.json
  {train,val}2017/      # image files that are mentioned in the corresponding json
```
### Expected dataset structure for [Pascal VOC detection](https://opendatalab.org.cn/OpenDataLab/PASCAL_VOC2012):
```
VOC2007/
  Annotations/ *.xml   # corresponding xml 
  ImageSets/Main/{train,text}.txt # train and val split file 
  JPEGImages/  *.jpg # image files
```
### Expected dataset structure for [Tusimple](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection):
```
tusimple_lane/
    lane_detection/
        clips/         # image files
        label_data_{0313,0531,...}.jsom  # json for lane label
```

### Expected dataset structure for [Culane](https://xingangpan.github.io/projects/CULane.html):
```
CULane/
  driver_*_frame/*MP4/ *.jpg *.lines.txt
  ...
  list/{train,val,text}.txt 
```

## Train Quantized model
Modify the dataset path configuration in the config file according to your dataset's location, and adjust the training parameters as needed.
### Pascal VOC
```
# yolov3-tiny 
python train_od.py -f configs/yolox_exp/yolov3tiny_voc_quant.py -d 1 -b 64 -o
# res34+yolov3
python train_od.py -f configs/yolox_exp/res34+yolov3_voc_quant.py -d 1 -b 64 -o
```
### COCO
```
# yolov3-tiny 
python train_od.py -f configs/yolox_exp/yolov3tiny_coco_quant.py -d 1 -b 32 -o
# res34+yolov3
python train_od.py -f configs/yolox_exp/res34+yolov3_coco_quant.py -d 1 -b 32 -o
```

### Tusimple
```
# resnet18
python train_ld.py --config configs/Tusimple/resnet18_condlane.py --work_name resnet18_condlane
# resnet34
python train_ld.py --config configs/Tusimple/resnet34_condlane.py --work_name resnet34_condlane
```

### Culane

```
# resnet18
python train_ld.py --config configs/CULane/resnet18_condlane.py --work_name resnet18_condlane
# resnet34
python train_ld.py --config configs/CULane/resnet34_condlane.py --work_name resnet34_condlane
```

## Eval with different neuron model
    --neuron    [tdIF, A2F] # A2F is IF neuron with delay
    --time_step int         # time step for inference
    --delay     int         # delay spike step

Partial model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1dyJb20076vaJ1G-gypzU4xLd0fttJX6p?usp=drive_link).

### Object detection
```
python eval_od.py -f configs/yolox_exp/yolov3tiny_voc_quant.py  -b 64 --neuron tdIF --time_step 8 --delay 3 --ckpt path_to/*ckpt.pth
```

### Lane line detection
```
python eval_ld.py --config configs/Tusimple/resnet18_condlane.py  --neuron tdIF --time_step 8 --delay 3 --load_from path_to/*ckpt.pth
```

## Citation
If our work help to your research, please cite our paper, thx.
```
@article{zhang2025ultra,
  title={Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection},
  author={Zhang, Chengjun and Zhang, Yuhao and Yang, Jie and Sawan, Mohamad},
  journal={arXiv preprint arXiv:2508.20392},
  year={2025}
}
```

## Thanks to these amazing projects:
- [spikingjelly](https://github.com/fangwei123456/spikingjelly)
- [ANN-SNN-QCFS](https://github.com/putshua/SNN_conversion_QCFS?tab=readme-ov-file)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [CondLaneNet](https://github.com/aliyun/conditional-lane-detection)
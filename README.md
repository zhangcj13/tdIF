# tdIF
The source code for paper: Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection

To view our paper, please refer: [Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection](https://arxiv.org/abs/2508.20392). The Supplementary Material is attached in this repo for reference.

## Method
![image](https://github.com/zhangcj13/tdIF/blob/main/images/overview.png)

<!-- ## Dependency
The denpendencies and versions are listed below: -->
<!-- ```
python                  3.8.13
CUDA                    11.7.1
torch                   1.13.0
torchaudio              0.13.0
torchvision             0.14.0
numpy                   1.23.3
urllib3                 1.26.12
spikingjelly            0.0.0.0.13
librosa                 0.7.1
Werkzeug                2.0.3
h5py                    3.10.0
``` -->

## Train Quantized model
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
tdIF and A2F (IF neuron with delay)
[Google Drive](https://drive.google.com/...)
### Object detection
```
python eval_od.py -f configs/yolox_exp/yolov3tiny_voc_quant.py  -b 64 --neuron tdIF --time_step 8 --delay 3 --ckpt /root/data1/ws/SNN_CV/YOLOX_outputs/tdIF_voc_yolov3tiny/yolov3tiny_voc_qcfs_ts255/best_ckpt.pth
```
### Lane line detection
```
python eval_ld.py --config configs/Tusimple/resnet18_condlane.py  --neuron tdIF --time_step 8 --delay 3 --load_from /root/data1/ws/SNN_CV/work_dirs/TuSimple/tdIF_resnet18_condlane_tusimple_qcfs_ts255/20241219_135725_lr_3e-04_b_96_ts_255/ckpt/best.pth
```

### Citation
If our work help to your research, please cite our paper, thx.
```
@article{zhang2025ultra,
  title={Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection},
  author={Zhang, Chengjun and Zhang, Yuhao and Yang, Jie and Sawan, Mohamad},
  journal={arXiv preprint arXiv:2508.20392},
  year={2025}
}
```

Thanks to：
- [spikingjelly](https://github.com/fangwei123456/spikingjelly)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [CondLaneNet](https://github.com/aliyun/conditional-lane-detection)
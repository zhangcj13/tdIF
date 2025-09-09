# tdIF
The source code for paper: Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection

To view our paper, please refer: [Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection](https://arxiv.org/abs/2508.20392). The Supplementary Material is attached in this repo for reference.

## Method
![image](https://github.com/zhangcj13/tdIF/blob/main/images/overview.png)

## Dependency
The denpendencies and versions are listed below:
```
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
```

## Reproduce
### Pascal VOC
```
python train.py 
```
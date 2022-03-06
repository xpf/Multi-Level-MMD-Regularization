# Multi-Level-MMD-Regularization (PyTorch)

<div align=center> <img src="./figures/cifar10_vgg11_blended.png"/> </div>

<div align=center> <img src="./figures/cifar10_vgg11_blended_mlmmdr_0.1_all.png"/> </div>

<div align=center> <img src="./figures/cifar10_vgg11_blended_mlmmdr_0.1_last.png"/> </div>

[Enhancing Backdoor Attacks with Multi-Level MMD Regularization]()

Pengfei Xia, Hongjing Niu, Ziqiang Li, and Bin Li.

>Abstract: *.*

## Training
```python
# Train VGG-11 on CIFAR-10 without ML-MMDR
python main.py --data_path your_path --data_name cifar10 --model_name vgg11 --mlmmdr_lamb 0

# Train VGG-11 on CIFAR-10 with ML-MMDR and lambda set to 0.1
python main.py --data_path your_path --data_name cifar10 --model_name vgg11 --mlmmdr_lamb 0.1 --mlmmdr_layer all

# Train VGG-11 on CIFAR-10 with SL-MMDR and lambda set to 0.1
python main.py --data_path your_path --data_name cifar10 --model_name vgg11 --mlmmdr_lamb 0.1 --mlmmdr_layer last
```

## Visualizing
```python
# Visualize VGG-11 trained on CIFAR-10 without ML-MMDR
python visualize.py --data_path your_path --data_name cifar10 --model_name vgg11 --mlmmdr_lamb 0

# Visualize VGG-11 trained on CIFAR-10 with ML-MMDR and lambda set to 0.1
python visualize.py --data_path your_path --data_name cifar10 --model_name vgg11 --mlmmdr_lamb 0.1 --mlmmdr_layer all

# Visualize VGG-11 trained on CIFAR-10 with SL-MMDR and lambda set to 0.1
python visualize.py --data_path your_path --data_name cifar10 --model_name vgg11 --mlmmdr_lamb 0.1 --mlmmdr_layer last
```

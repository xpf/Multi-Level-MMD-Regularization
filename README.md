# Multi-Level-MMD-Regularization (PyTorch)

<div align=center> <img src="./figures/cifar10_vgg11_blended.png" width="70%" height="70%"/> </div>

<div align=center> <img src="./figures/cifar10_vgg11_blended_mlmmdr_0.1_all.png" width="70%" height="70%"/> </div>

<div align=center> <img src="./figures/cifar10_vgg11_blended_mlmmdr_0.1_last.png" width="70%" height="70%"/> </div>

[Enhancing Backdoor Attacks with Multi-Level MMD Regularization]()

Pengfei Xia, Hongjing Niu, Ziqiang Li, and Bin Li.

>Abstract: *While Deep Neural Networks (DNNs) excel in many tasks, it comes at the cost of significant training resources. To save costs, it has become a common practice to collect data from the Internet or hire a third party to train a model. Unfortunately, recent studies have shown that these operations provide a viable path for injecting a hidden backdoor into a DNN. An infected model behaves normally on benign inputs, whereas its predictions are forced to an attacker-specific target on malicious inputs. Many defense methods have been developed to detect malicious samples. Their common assumption is that the latent representations of clean and malicious samples extracted by the infected model belong to two different distributions. However, a comprehensive study of the distributional differences is lacking. In this paper, we focus on it and investigate three questions: 1) What are the characteristics of the distributional differences? 2) How can they be effectively reduced? 3) What impact does this reduction have on difference-based defense methods? Our work is carried out based on the above questions. First, by introducing Maximum Mean Discrepancy (MMD), Energy Distance (ED), and Sliced Wasserstein Distance (SWD) as the metrics, we identify that the distributional differences of multi-level representations are all large, not just at the highest level. Then, we propose ML-MMDR, a reduction method that adds Multi-Level MMD Regularization to the loss during the training of a backdoored model to fully reduce the differences. Finally, three typical difference-based defense methods are tested. Across all two datasets and four DNN architectures, the F1 scores of these methods drop from 90%-100% on the regularly trained backdoored models to 60%-70% on the models trained with ML-MMDR. These results indicate that the proposed regularization can enhance the stealthiness of existing backdoor attack methods.*

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

# Multi-Level-MMD-Regularization (PyTorch)

<div align=center> <img src="./figures/cifar10_vgg11.png"/> </div>

[Enhancing Backdoor Attacks with Multi-Level MMD Regularization]()

Pengfei Xia, Hongjing Niu, Ziqiang Li, and Bin Li

>Abstract: *While Deep Neural Networks (DNNs) excel in many tasks, it comes at the cost of significant training resources. To save the cost, it has become a common practice to collect data from the Internet or hire a third party to train a model. Unfortunately, recent studies have shown that these operations provide a viable pathway for injecting a hidden backdoor into a DNN. An infected model behaves normally on benign inputs, whereas on malicious inputs, its predictions will be forced to an attacker-specific target. Many defense methods have been developed to detect malicious samples, and their common assumption is that the latent representations of clean and malicious samples extracted by the infected model belong to two different distributions. However, a comprehensive study on the distributional differences is lacking. In this paper, we focus on it and investigate three questions: 1) What are the characteristics of the distributional differences? 2) How to effectively reduce them? 3) What impact does this reduction have on difference-based defense methods? Our work is carried out on the above questions. First, by introducing Maximum Mean Discrepancy (MMD), Energy Distance (ED), and Sliced Wasserstein Distance (SWD) as the metrics, we identify that the distributional differences of multi-level representations are all large, not just at the highest level. Then, we propose ML-MMDR, a reduction method by adding a Multi-Level MMD Regularization to the loss during training a backdoored model to fully reduce the differences. Last, three typical difference-based defense methods are tested. Across all two datasets and four DNN architectures, the F1 scores of these methods drop from 90%-100% on the regularly trained backdoored models to 60%-70% on the models trained with ML-MMDR. These results indicate that the proposed regularization can enhance the stealthiness of existing backdoor attack methods.*

## Training



# In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learn

Implementation of [In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning](https://arxiv.org/abs/2101.06329). The paper presents an uncertainty-aware pseudo-label selection framework for semi-supervised learning which greatly reduces the noise introduced by the pseudo-labeling process.

The recent research in semi-supervised learning (SSL) is mostly dominated by consistency regularization based methods which achieve strong performance. However, they heavily rely on domain-specific data augmentations, which are not easy to generate for all data modalities. Pseudo-labeling (PL) is a general SSL approach that does not have this constraint but performs relatively poorly in its original formulation. We argue that PL underperforms due to the erroneous high confidence predictions from poorly calibrated models; these predictions generate many incorrect pseudo-labels, leading to noisy training. We propose an uncertainty-aware pseudo-label selection (UPS) framework which improves pseudo labeling accuracy by drastically reducing the amount of noise encountered in the training process. Furthermore, UPS generalizes the pseudo-labeling process, allowing for the creation of negative pseudo-labels; these negative pseudo-labels can be used for multi-label classification as well as negative learning to improve the single-label classification. We achieve strong performance when compared to recent SSL methods on the CIFAR-10 and CIFAR-100 datasets. Also, we demonstrate the versatility of our method on the video dataset UCF-101 and the multi-label dataset Pascal VOC.

This repository is implemented using PyTorch and it includes code for running the SSL experiments on CIFAR-10 and CIFAR-100 datasets.

## Dependencies
This code requires the following:

* Python >= 3.6
* numpy==1.16.2
* Pillow==5.4.1
* scikit-learn==0.21.1
* scipy==1.2.1
* torch==1.7.0
* torchvision==0.8.1
* tqdm==4.36.1
* tensorboardx==1.7
* tensorboard==1.13.1

run `pip3 install -r requirements.txt` to install all the dependencies.

## Training
```shell
# For CIFAR10 4000 Labels
python3 train-cifar.py --dataset "cifar10" --n-lbl 4000 --class-blnc 7 --split-txt "run1" --arch "cnn13"

# For CIFAR10 1000 Labels
python3 train-cifar.py --dataset "cifar10" --n-lbl 1000 --class-blnc 7 --split-txt "run1" --arch "cnn13"

# For CIFAR100 10000 Labels
python3 train-cifar.py --dataset "cifar100" --n-lbl 10000 --class-blnc 1 --split-txt "run1" --arch "cnn13"

# For CIFAR100 4000 Labels
python3 train-cifar.py --dataset "cifar100" --n-lbl 4000 --class-blnc 1 --split-txt "run1" --arch "cnn13"
```

## Citation
```
@inproceedings{rizve2021in,
title={In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning},
author={Mamshad Nayeem Rizve and Kevin Duarte and Yogesh S Rawat and Mubarak Shah},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=-ODN6SbiUU}
}
```

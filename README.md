# In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning

Official implementation of [In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning](https://arxiv.org/abs/2101.06329). The paper presents an uncertainty-aware pseudo-label selection framework for semi-supervised learning which greatly reduces the noise introduced by the pseudo-labeling process.

The recent research in semi-supervised learning (SSL) is mostly dominated by consistency regularization based methods which achieve strong performance. However, they heavily rely on domain-specific data augmentations, which are not easy to generate for all data modalities. Pseudo-labeling (PL) is a general SSL approach that does not have this constraint but performs relatively poorly in its original formulation. We argue that PL underperforms due to the erroneous high confidence predictions from poorly calibrated models; these predictions generate many incorrect pseudo-labels, leading to noisy training. We propose an uncertainty-aware pseudo-label selection (UPS) framework which improves pseudo labeling accuracy by drastically reducing the amount of noise encountered in the training process. Furthermore, UPS generalizes the pseudo-labeling process, allowing for the creation of negative pseudo-labels; these negative pseudo-labels can be used for multi-label classification as well as negative learning to improve the single-label classification. We achieve strong performance when compared to recent SSL methods on the CIFAR-10 and CIFAR-100 datasets. Also, we demonstrate the versatility of our method on the video dataset UCF-101 and the multi-label dataset Pascal VOC.

This repository is implemented using PyTorch and it includes code for running the SSL experiments on CIFAR-10 and CIFAR-100 datasets.

<p align="center">
  <img src="/figures/conceptual-1.png" width="500">
</p>
<p>
  <em>Approach Overview: Shapes represent different transformations and colors represent different classes. While the invariant features provide better discrimination, the equivariant features help us learn the internal structure of the data manifold. These complimentary representations help us generalize better to new tasks with only a few training samples. By jointly leveraging the strengths of equivariant and invariant features, our method achieves significant improvement over baseline (bottom row).</em>
</p>

<p align="center">
  <img src="/figures/training.png" width="800">
</p>
<p>
  <em>Network Architecture during Training: A series of transformed inputs (transformed by applying transformations T1...TM) are
provided to a shared feature extractor fΘ. The resulting embedding is forwarded to three parallel heads fΦ, fΘ and fΩ that focus on
learning equivariant features, discriminative class boundaries, and invariant features, respectively. The resulting output representations are
distilled from an old copy of the model (teacher model on the right) across multiple-heads to further improve the encoded representations.
Notably, a dedicated memory bank of negative samples helps stabilize our invariant contrastive learning.</em>
</p>

## Dependencies

# Adaptive-Gradient-Clipping
This repository provides a minimal implementation of adaptive gradient clipping (AGC) (as proposed in High-Performance Large-Scale Image Recognition Without Normalization<sup>1</sup>) in TensorFlow 2. The paper attributes AGC as a crucial component in order to train deep neural networks without batch normalization<sup>2</sup>. Readers are encouraged to consult the paper to understand why one might want to train networks without batch normalization given its paramount success. 

My goal with this repository is to be able to _quickly_ train shallow networks with and without AGC. Therefore, I provide two Colab Notebooks which I discuss below. 

## About the notebooks

* `AGC.ipynb`: Demonstrates training of a shallow network (only 0.002117 million parameters) with AGC. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayakpaul/Adaptive-Gradient-Clipping/blob/main/AGC.ipynb)
* `BatchNorm.ipynb`: Demonstrates training of a shallow network (only 0.002309 million parameters) with batch normalization. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayakpaul/Adaptive-Gradient-Clipping/blob/main/BatchNorm.ipynb)

Both of these notebooks are end-to-end executable on Google Colab. Furthermore, they utilize the free TPUs (TPUv2-8) Google Colab provides allowing readers to experiment very quickly. 

## Findings

Before moving to the findings, please be aware of the following things:
* The network I have used in order to demonstrate the results is extremely shallow.
* The network is a mini VGG<sup>3</sup> style network whereas the original paper focuses on ResNet<sup>4</sup> style architectures. 
* The dataset (**flowers dataset**) I experimented with consists of ~3500 samples.
* I clipped gradients of all the layers whereas in the original paper final linear layer wasn't clipped (refer to Section 4.1 of the original paper).

By comparing the training progress of two networks (trained with and without AGC), we see that with AGC network training is more stabilized.

Batch Normalization            |  AGC
:-------------------------:|:-------------------------:
![](https://i.ibb.co/4KXkMDH/image.png)  |  ![](https://i.ibb.co/74Xdsbj/image.png)

In the table below, I summarize results of the two aforementioned notebooks - 

|                            | Number of Parameters (million) | Final Validation Accuracy (%) | Training Time (seconds) |
|:--------------------------:|:------------------------------:|:-----------------------------:|:-----------------------:|
|     Batch Normalization    |            0.002309            |             54.67             |          2.7209         |
| Adaptive Gradient Clipping |            0.002117            |               52              |          2.6145         |

For these experiments, I used a batch size of 512 each batch having a shape of `(512, 96, 96, 3)` and a clipping factor of 0.01 (applicable only for AGC).

These results SHOULD NOT be treated as conclusive. For details related to training configuration (i.e. network depth, learning rate, etc.) please refer to the notebooks. 

## Citations

[1] Brock, Andrew, et al. “High-Performance Large-Scale Image Recognition Without Normalization.” ArXiv:2102.06171 [Cs, Stat], Feb. 2021. arXiv.org, http://arxiv.org/abs/2102.06171.

[2] Ioffe, Sergey, and Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” ArXiv:1502.03167 [Cs], Mar. 2015. arXiv.org, http://arxiv.org/abs/1502.03167.

[3] Simonyan, Karen, and Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” ArXiv:1409.1556 [Cs], Apr. 2015. arXiv.org, http://arxiv.org/abs/1409.1556.

[4] He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” ArXiv:1512.03385 [Cs], Dec. 2015. arXiv.org, http://arxiv.org/abs/1512.03385.

## Acknowledgements

I referred to the following resources during experimentation:
* [Original JAX implementation of AGC](https://github.com/deepmind/deepmind-research/blob/master/nfnets/agc_optax.py) provided by the authors. 
* [Ross Wightman's implementation og AGC](https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py).
* [Fast and Lean Data Science materials](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/fast-and-lean-data-science) provided by GCP. 

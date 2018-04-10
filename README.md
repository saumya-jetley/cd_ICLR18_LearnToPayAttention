# ActiveAttention
Active attention in classification networks: Attention that is optimised at the time of model training.

Installations:
- Torch (can be installed with sudo and out-of-the-box as long as all the dependencies are installed on the machine.)
- optnet package - to optimise the network
- iterm package
- image package
- setup cudnn and provide path (LD_LIBRARY_PATH or export CUDNN_PATH='/cudart.so.5')


Sample Run:
 - Enter VGGNet folder
 - Refer to the .sh scripts for the train and test settings for the different architectures and datasets



Note: This code is built on top of:
1. https://github.com/szagoruyko/cifar.torch
In particular the training parameters for VGG network can be found here - https://github.com/szagoruyko/cifar.torch/blob/master/train.lua 

2. https://github.com/szagoruyko/wide-residual-networks/tree/fp16

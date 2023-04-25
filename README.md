# CS579-HW1
Replication of three Convolutional Neural Networks(CNNs) architecture: [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf), [VGG16](https://arxiv.org/abs/1409.1556), and [ResNet18](https://arxiv.org/abs/1512.03385). All three models will be train on two image classification datasets: [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST) and [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10). Models are implemented with PyTorch. Minor changes to the architechtures will be incorporated for easier implementation and better training.

## LeNet
![LeNet Architecture](https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/lenet-5.png)
## VGG16
![VGG16 Architecture](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*_Lg1i7wv1pLpzp2F4MLrvw.png)
## ResNet18
![ResNet18 Architecture](https://miro.medium.com/v2/resize:fit:640/format:webp/1*kBlZtheCjJiA3F1e0IurCw.png)

## Results
### MNIST
| Architecture | Accuracy |
| :----------: | :------: |
| LeNet        | 99.08%   |
| VGG16        | 99.46%   |
| ResNet18     | 99.65%   |
### CIFAR10
| Architecture | Accuracy |
| :----------: | :------: |
| LeNet        | 69.18%   |
| VGG16        | 83.44%   |
| ResNet18     | 90.93%   |


## Datasets
`MNIST`: http://yann.lecun.com/exdb/mnist/  
`CIFAR10`: https://www.cs.toronto.edu/~kriz/cifar.html

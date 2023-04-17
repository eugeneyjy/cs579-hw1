import torch
import torch.nn as nn

from torchvision import transforms

from models.lenet.arch import LeNet
from models.vgg16.arch import VGG16
from models.resnet18.arch import ResNet18

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(args, num_classes):
    # Set number of input channels based on dataset
    channels = 1
    if args.dataset == 'cifar10':
        channels = 3

    if args.arch == 'lenet':
        return LeNet(num_classes, channels).to(device), (32, 32)
    elif args.arch == 'vgg16':
        return VGG16(num_classes, channels).to(device), (224, 224)
    elif args.arch == 'resnet18':
        return ResNet18(num_classes, channels).to(device), (224, 224)

def get_criterion(loss):
    if loss == 'crossEntropy':
        return nn.CrossEntropyLoss()
    
def get_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return torch.optim.Adam(parameters, args.lr, amsgrad=True)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(parameters, args.lr, 0.9)
    
def get_transform(args, input_size):
    all_transforms = [transforms.Resize(input_size)]
    
    if 'horizontal_flip' in args and args.horizontal_flip:
        all_transforms.append(transforms.RandomHorizontalFlip())
    if 'rotation' in args and args.rotation:
        all_transforms.append(transforms.RandomRotation(20))
    
    all_transforms.append(transforms.ToTensor())
    
    if args.dataset == 'mnist':
        all_transforms.append(transforms.Normalize(mean = (0.13066062,), std = (0.30810776,)))
    elif args.dataset == 'cifar10':
        all_transforms.append(transforms.Normalize(mean = (0.4914009,0.48215896,0.4465308), std = (0.24703279,0.24348423,0.26158753)))
    
    return transforms.Compose(all_transforms)

def get_plot_title(args):
    if (args.arch == 'lenet'):
        model_name = 'LeNet'
    elif (args.arch == 'vgg16'):
        model_name = 'VGG16'
    elif (args.arch == 'resnet18'):
        model_name = 'ResNet18'

    if (args.dataset == 'mnist'):
        dataset_name = 'MNIST'
    elif (args.dataset == 'cifar10'):
        dataset_name = 'CIFAR10'

    return f'{model_name} Trained On {dataset_name}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
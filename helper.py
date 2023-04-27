import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
        model = LeNet(num_classes, channels).to(device)
        input_size = (32, 32)
    elif args.arch == 'vgg16':
        model = VGG16(num_classes, channels).to(device)
        input_size = (224, 224)
    elif args.arch == 'resnet18':
        if 'dropout' in args:
            model = ResNet18(num_classes, channels, args.dropout).to(device)
        else:
            model = ResNet18(num_classes, channels).to(device)
        input_size = (224, 224)
    
    if 'path' in args and args.path:
        checkpoint = torch.load(args.path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, input_size

def get_criterion(loss):
    if loss == 'crossEntropy':
        return nn.CrossEntropyLoss()
    
def get_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return torch.optim.Adam(parameters, args.lr, amsgrad=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(parameters, args.lr, 0.9)
    
def get_transform(args, input_size):
    all_transforms = [transforms.Resize(input_size)]
    
    if 'horizontal_flip' in args and args.horizontal_flip:
        all_transforms.append(transforms.RandomHorizontalFlip())
    if 'rotation' in args and args.rotation:
        all_transforms.append(transforms.RandomRotation(20))
    
    all_transforms.append(transforms.ToTensor())
    
    if 'normalize' in args and args.normalize:
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

def show_image(x, y):
    img = x.detach().numpy().transpose(1, 2, 0)
    print(img.shape)
    plt.imshow(img)
    plt.title(y)
    plt.show()

# function adopted from https://adversarial-ml-tutorial.org/adversarial_examples/
def plot_images(X,y,yhat,M,N):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N*2,M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i*N+j].cpu().numpy().transpose(1, 2, 0))
            title = ax[i][j].set_title("Ori: {}\nPred: {}".format(map_label(y[i*N+j]), map_label(yhat[i*N+j])))
            plt.setp(title, color=('g' if yhat[i*N+j] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    plt.show()

def map_label(number):
    if number == 0:
        return 'airplane'
    elif number == 1:
        return 'automobile'
    elif number == 2:
        return 'bird'
    elif number == 3:
        return 'cat'
    elif number == 4:
        return 'deer'
    elif number == 5:
        return 'dog'
    elif number == 6:
        return 'frog'
    elif number == 7:
        return 'horse'
    elif number == 8:
        return 'ship'
    elif number == 9:
        return 'truck'

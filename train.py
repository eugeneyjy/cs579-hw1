import argparse

import torch
import torch.nn as nn

from torchvision import transforms

from models.lenet.arch import LeNet
from models.vgg16.arch import VGG16
from datasets import data_loader
from valid import validation_metrics
from path import MODEL_DIR, REPORT_DIR

from tqdm import tqdm
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

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

def get_criterion(loss):
    if loss == 'crossEntropy':
        return nn.CrossEntropyLoss()
    
def get_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return torch.optim.Adam(parameters, args.lr, amsgrad=True)

def train_model(model, train_loader, val_loader, optimizer, criterion, args):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    min_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            _, pred_label = torch.max(pred, 1)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            correct += (pred_label == target).sum()
            total += len(data)

            if i % 64 == 0:
                logging.info("Epoch [%d/%d] || Step [%d/%d] || Loss: [%f] || Acc: [%f]" % 
                             (epoch, args.epochs, i, len(train_loader), sum_loss/total, correct/total))

        train_loss, train_acc = sum_loss/total, correct/total

        logging.info("calculating validation metrics")
        val_loss, val_acc = validation_metrics(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc.cpu())
        val_losses.append(val_loss)
        val_accs.append(val_acc.cpu())

        logging.info("Epoch %d train loss %f, train acc %.3f, val loss %f, val acc %.3f" % 
                    (epoch, train_loss, train_acc, val_loss, val_acc))

        if not args.no_save:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                logging.info("Saving better model")
                save_model(epoch, model, optimizer, args)

    return train_losses, train_accs, val_losses, val_accs

def save_model(epoch, model, optimizer, args):
    path_name = f'{MODEL_DIR}/{args.arch}/{args.arch}-{args.dataset}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        path_name
    )

def get_plot_title(args):
    if (args.arch == 'lenet'):
        model_name = 'LeNet'
    elif (args.arch == 'vgg16'):
        model_name = 'VGG16'

    if (args.dataset == 'mnist'):
        dataset_name = 'MNIST'
    elif (args.dataset == 'cifar10'):
        dataset_name = 'CIFAR10'

    return f'{model_name} Trained On {dataset_name}'

def plot_results(args, results):
    # result: (train_losses, train_accs, val_losses, val_accs)
    epochs = range(len(results[0]))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    fig.suptitle(get_plot_title(args))

    # Plot training and validation losses 
    ax1.set_title('Model Loss')
    ax1.plot(epochs, results[0], label='training')
    ax1.plot(epochs, results[2], label='validation')
    ax1.legend(loc='upper left')

    # Plot training and validation accuracies
    ax2.set_title('Model Accuracy')
    ax2.plot(epochs, results[1], label='training')
    ax2.plot(epochs, results[3], label='validation')
    ax2.legend(loc='upper left')

    plt.savefig(f'{REPORT_DIR}/{args.arch}-{args.dataset}.png')
    plt.show()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='lenet', metavar='lenet',
                        help='model architecture to train on (default: lenet)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='64',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='10',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='0.001',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='mnist',
                        help='dataset to train on (default: mnist)')
    parser.add_argument('--loss', type=str, default='crossEntropy', metavar='crossEntropy',
                        help='loss function (default= crossEntropy)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='adam',
                        help='optimization algorithm (default: adam)')
    parser.add_argument('--no-save', action='store_true',
                        help='specify if want the trained model be saved (default: True)')
    parser.add_argument('--seed', type=int, default=113, metavar='113',
                        help='specify seed for random (default: 113)')
                    
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    model, input_size = get_model(args, 10)

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])

    train_loader, val_loader = data_loader(args.dataset, args.batch_size, transform=transform)
    criterion = get_criterion(args.loss)
    optimizer = get_optimizer(args, model.parameters())

    results = train_model(model, train_loader, val_loader, optimizer, criterion, args)

    plot_results(args, results)

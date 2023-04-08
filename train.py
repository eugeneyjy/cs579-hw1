import argparse

import torch
import torch.nn as nn

from torchvision import transforms

from models.lenet.arch import LeNet
from datasets import data_loader
from valid import validation_metrics

from tqdm import tqdm

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(arch, num_classes):
    if arch == 'lenet':
        return LeNet(num_classes).to(device)

def get_criterion(loss):
    if loss == 'crossEntropy':
        return nn.CrossEntropyLoss()
    
def get_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return torch.optim.Adam(parameters, args.lr, amsgrad=True)

def train_model(model, train_loader, val_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
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

            sum_loss += loss
            correct += (pred_label == target).sum()
            total += len(data)

        val_loss, val_acc = validation_metrics(model, val_loader, criterion)

        logging.info("epoch %d train loss %f, train acc %.3f, val loss %f, val acc %.3f" % 
                    (epoch, sum_loss/total, correct/total, val_loss, val_acc))


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
                        help='optimization algorithm (default= adam)')
        
    args = parser.parse_args()
    print(args)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_loader, val_loader = data_loader(args.dataset, args.batch_size, transform=transform)
    model = get_model(args.arch, 10)
    criterion = get_criterion(args.loss)
    optimizer = get_optimizer(args, model.parameters())

    train_model(model, train_loader, val_loader, args.epochs, optimizer, criterion)

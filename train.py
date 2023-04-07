import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from models.lenet.arch import LeNet
from datasets import data_loader

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    args = parser.parse_args()
    print(args)

    model = LeNet(10).to(device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_loader, _ = data_loader(args.dataset, args.batch_size, transform=transform)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

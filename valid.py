import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datasets import data_loader, get_dataset
from torch.utils.data import DataLoader
from helper import get_model, get_transform, get_criterion, show_image, map_label
from path import DATASETS_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation_metrics(model, loader, criterion):
    model.eval()
    sum_loss = 0
    correct = 0
    total = 0
    for i, (data, target) in tqdm(enumerate(loader), total=len(loader)):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        _, pred_label = torch.max(pred, 1)

        loss = criterion(pred, target)
        print(map_label(pred_label[0]))

        sum_loss += loss.item()
        correct += (pred_label == target).sum().item()
        total += len(data)

    return sum_loss/total, correct/total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='lenet', metavar='lenet',
                        help='model architecture to evaluate on (default: lenet)')
    parser.add_argument('--path', type=str, help='path that the model get save on')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='mnist',
                        help='dataset to evaluate on (default: mnist)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='64',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--loss', type=str, default='crossEntropy', metavar='crossEntropy',
                        help='loss function (default= crossEntropy)')
    parser.add_argument('--subsample', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9], metavar='0 1 2 3 4 5 6 7 8 9', 
                        help='specify the specific class we want to get from the dataset (default: 1 - 10)')
    parser.add_argument('--targetId', type=int, default=-1, metavar='-1',
                        help='index of the only test image we want to check for')
    

    args = parser.parse_args()

    model, input_size = get_model(args, len(args.subsample))

    model.eval()

    transform = get_transform(args, input_size)
    criterion = get_criterion(args.loss)

    # test_loader = data_loader(args.dataset, args.batch_size, transform=transform, train=False, subsample=args.subsample)
    test_dataset = get_dataset(args.dataset, transform, train=False, subsample=args.subsample)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.targetId != -1:
    # test target image accuracy
        test_dataset = get_dataset(args.dataset, transform, train=False)
        test_dataset.targets = np.array(test_dataset.targets)
        test_dataset.data = np.array(test_dataset.data)
        class_inds = (test_dataset.targets==6)
        test_dataset.targets = test_dataset.targets[class_inds]
        test_dataset.data = test_dataset.data[class_inds]
        test_dataset.targets = test_dataset.targets[[args.targetId]]
        test_dataset.data = test_dataset.data[[args.targetId]]
        test_loader = DataLoader(dataset=test_dataset)
        show_image(next(iter(test_loader))[0].squeeze(), map_label(6), args.targetId)

    print(validation_metrics(model, test_loader, criterion))
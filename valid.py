import torch
import argparse
from tqdm import tqdm
from datasets import data_loader
from helper import get_model, get_transform, get_criterion

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
    

    args = parser.parse_args()

    model, input_size = get_model(args, 10)

    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = get_transform(args, input_size)
    criterion = get_criterion(args.loss)

    test_loader = data_loader(args.dataset, args.batch_size, transform=transform, train=False)
    print(validation_metrics(model, test_loader, criterion))
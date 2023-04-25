import argparse
import sys

import torch

from datasets import data_loader, batch_mean_and_std
from valid import validation_metrics
from path import get_report_dir
from helper import get_model, get_criterion, get_optimizer, get_transform, get_plot_title, set_seed

from tqdm import tqdm
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            correct += (pred_label == target).sum().item()
            total += len(data)

            if i % 64 == 0:
                # print(f'Prediction: {pred_label}')
                # print(f'Target:     {target}')
                logging.info("Epoch [%d/%d] || Step [%d/%d] || Loss: [%f] || Acc: [%f]" % 
                             (epoch+1, args.epochs, i, len(train_loader), sum_loss/total, correct/total))

        train_loss, train_acc = sum_loss/total, correct/total

        if val_loader:
            logging.info("calculating validation metrics")
            val_loss, val_acc = validation_metrics(model, val_loader, criterion)
        else:
            val_loss, val_acc = 0.0, 0.0

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logging.info("Epoch %d train loss %f, train acc %.3f, val loss %f, val acc %.3f" % 
                    (epoch+1, train_loss, train_acc, val_loss, val_acc))

        if not args.no_save:
            logging.info("Saving model")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_model(epoch+1, model, optimizer, True, args)
            save_model(epoch+1, model, optimizer, False, args)

    return train_losses, train_accs, val_losses, val_accs

def save_model(epoch, model, optimizer, best, args):
    if best:
        path_name = f'{get_report_dir()}/{args.arch}-{args.dataset}-best.pth'
    else:
        path_name = f'{get_report_dir()}/{args.arch}-{args.dataset}-last.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        path_name
    )

def plot_results(args, results):
    # result: (train_losses, train_accs, val_losses, val_accs)
    epochs = range(1, len(results[0])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    fig.suptitle(get_plot_title(args))

    # Plot training and validation losses 
    ax1.set_title('Model Loss')
    ax1.plot(epochs, results[0], label='training')
    ax1.plot(epochs, results[2], label='validation')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')

    # Plot training and validation accuracies
    ax2.set_title('Model Accuracy')
    ax2.plot(epochs, results[1], label='training')
    ax2.plot(epochs, results[3], label='validation')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')

    plt.savefig(f'{get_report_dir()}/plots.png')
    # plt.show()

def save_hyperparameters(args):
    with open(f'{get_report_dir()}/hyperparameters.txt', 'w') as f:
        f.write(f'{args}')

def save_results(results):
    train_loss = results[0]
    train_acc = results[1]
    val_loss = results[2]
    val_acc = results[3]
    epoch = len(train_loss)

    with open(f'{get_report_dir()}/results.csv', 'w') as f:
        f.write('train_loss,train_acc,val_loss,val_acc\n')
        for i in range(epoch):
            f.write(f'{train_loss[i]},{train_acc[i]},{val_loss[i]},{val_acc[i]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='lenet', metavar='lenet',
                        help='model architecture to train on (default: lenet)')
    parser.add_argument('--epochs', type=int, default=10, metavar='10',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='64',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='0.001',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='mnist',
                        help='dataset to train on (default: mnist)')
    parser.add_argument('--loss', type=str, default='crossEntropy', metavar='crossEntropy',
                        help='loss function (default= crossEntropy)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='adam',
                        help='optimization algorithm (default: adam)')
    parser.add_argument('--full-training', action='store_true',
                        help='specify if want to train with full training data without having validation data (default: False)')
    parser.add_argument('--no-save', action='store_true',
                        help='specify if do not want the trained model be saved (default: False)')
    parser.add_argument('--seed', type=int, default=113, metavar='113',
                        help='specify seed for random (default: 113)')
    parser.add_argument('--horizontal-flip', action='store_true',
                        help='specify if want the image data be flip horizontally at random (default: False)')
    parser.add_argument('--rotation', action='store_true',
                        help='specify if want the image data be rotate (-90, 90) degree at random (default: False)')
    parser.add_argument('--normalize', action='store_true',
                        help='specify if want to normalize data before train (default: False)')
                    
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    model, input_size = get_model(args, 10)

    transform = get_transform(args, input_size)

    if args.full_training:
        train_loader = data_loader(args.dataset, args.batch_size, transform=transform, valid_size=0)
        val_loader = data_loader(args.dataset, args.batch_size, transform=transform, train=False)
    else:
        train_loader, val_loader = data_loader(args.dataset, args.batch_size, transform=transform)
    criterion = get_criterion(args.loss)
    optimizer = get_optimizer(args, model.parameters())

    logging.info("Training on %s" % (device))

    save_hyperparameters(args)
    
    results = train_model(model, train_loader, val_loader, optimizer, criterion, args)

    plot_results(args, results)
    save_results(results)

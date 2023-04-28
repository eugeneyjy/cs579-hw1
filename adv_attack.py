import argparse
import torch
import matplotlib.pyplot as plt

from helper import get_model, get_criterion, get_transform, set_seed, show_image, map_label, plot_images,device
from path import get_report_dir, create_mnist_label_dir
from datasets import data_loader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


def PGD(x, y, model, loss, niter=5, epsilon=0.03, stepsize=2/255, randinit=True):
    # make model train mode because of some weird behavior
    # without this PGD is weird that it drive resnet accuracy to 0 even for 1 iteration
    model.train()

    x, y = x.to(device), y.to(device)
    adv_imgs = x.clone().to(device).detach()
    if randinit:
        adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-epsilon, epsilon)
        adv_imgs = torch.clamp(adv_imgs, 0, 1).detach_()

    for _ in range(niter):
        adv_imgs.requires_grad_()
        pred = model(adv_imgs)
        cost = loss(pred, y)
        cost.backward()
        adv_imgs = adv_imgs + stepsize*adv_imgs.grad.sign()
        delta = torch.clamp(adv_imgs - x, -epsilon, epsilon)
        adv_imgs = torch.clamp(x + delta, 0, 1).detach_()
            
    return (adv_imgs.detach(), y)

def calc_adv_acc(model, test_loader, criterion, args):
    sum_loss = 0
    correct = 0
    total = 0
    for i, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        data, target = data.to(device), target.to(device)
        if not args.clean:
            data, _ = PGD(data, target, model, **get_PGD_kwargs(args))
        model.eval()
        pred = model(data)
        _, pred_label = torch.max(pred, 1)
        # plot_images(data, target, pred_label, 2, 5)

        loss = criterion(pred, target)

        sum_loss += loss.item()
        correct += (pred_label == target).sum().item()
        total += len(data)

    print(f'loss: {sum_loss/total}, acc: {correct/total}')
    print(f'correct: {correct}')
    return sum_loss/total, correct/total

def niter_vs_acc(model, test_loader, criterion, args):
    losses = []
    accs = []
    niters = [1,2,3,4,5,10,20,30,40,80,100]
    for i in niters:
        args.niter = i
        print(args)
        loss, acc = calc_adv_acc(model, test_loader, criterion, args)
        losses.append(loss)
        accs.append(acc)
        print(f'loss:{loss}, acc:{acc}')
    
    plt.plot(niters, accs)
    plt.title(f'{args.arch}-{args.dataset} PGD')
    plt.xlabel("# iterations")
    plt.ylabel("Accuracy")
    plt.savefig(f'{get_report_dir()}/niters_vs_accs.png')
    
    with open(f'{get_report_dir()}/results.csv', 'w') as f:
        f.write('niters,accs,loss\n')
        for i in range(len(accs)):
            f.write(f'{niters[i]},{accs[i]},{losses[i]}\n')

def epsilon_vs_acc(model, test_loader, criterion, args):
    losses = []
    accs = []
    epsilon = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    for i in epsilon:
        args.epsilon = i
        print(args)
        loss, acc = calc_adv_acc(model, test_loader, criterion, args)
        losses.append(loss)
        accs.append(acc)
        print(f'loss:{loss}, acc:{acc}')
    
    plt.plot(epsilon, accs)
    plt.title(f'{args.arch}-{args.dataset} PGD')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(f'{get_report_dir()}/epsilon_vs_accs.png')

    with open(f'{get_report_dir()}/results.csv', 'w') as f:
        f.write('epsilon,accs,loss\n')
        for i in range(len(accs)):
            f.write(f'{epsilon[i]},{accs[i]},{losses[i]}\n')

def craft_adv_exp(model, loader, args):
    x, y = next(iter(loader))
    adv_exps, _ = PGD(x, y, model, **get_PGD_kwargs(args))
    dataset = []
    for i in range(10):
        save_image(adv_exps[i, :, :, :], f'{get_report_dir()}/{i}.png')
        dataset.append([adv_exps[i], y[i]])
    return dataset

def get_PGD_kwargs(args):
    kwargs = {
        'loss': get_criterion(args.loss),
        'niter': args.niter,
        'epsilon': args.epsilon,
        'stepsize': args.stepsize,
        'randinit': args.randinit,
    }

    return kwargs

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
    parser.add_argument('--niter', type=int, default=5, metavar=5,
                        help='number of iteration in PGD (default= 5)')
    parser.add_argument('--epsilon', type=float, default=0.3, metavar=0.3,
                        help='epsilon value in PGD (default= 0.3)')
    parser.add_argument('--stepsize', type=float, default=2/255, metavar=2/255,
                        help='stepsize in PGD (default= 2/255)')  
    parser.add_argument('--randinit', action='store_true',
                        help='specify if start at random location for PGD (default= False)')
    parser.add_argument('--clean', action='store_true',
                        help='specify if want accuracy of clean test sample accuracy (default= False)')

    args = parser.parse_args()
    print(args)
    set_seed(113)

    model, input = get_model(args, 10)
    transform = get_transform(args, input)
    criterion = get_criterion(args.loss)
    test_loader = data_loader(args.dataset, 32, transform, train=False)

    loss, acc = calc_adv_acc(model, test_loader, criterion, args)
    # niter_vs_acc(model, test_loader, criterion, args)
    # epsilon_vs_acc(model, test_loader, criterion, args)
    # dataset = craft_adv_exp(model, test_loader, args)
    # test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=10)
    # loss, acc = calc_adv_acc(model, test_loader, criterion, args)

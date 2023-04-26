import argparse
import torch 

from helper import get_model, get_criterion, get_transform, set_seed, show_image, map_label, device
from path import get_report_dir, create_mnist_label_dir
from datasets import data_loader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


def PGD(x, y, model, loss, niter=5, epsilon=0.03, stepsize=2/255, randint=True):
    x, y = x.to(device), y.to(device)
    adv_imgs = x.clone().to(device)
    if randint:
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
    
    return (adv_imgs, y)


# def craft_adv_exp(model, loader, dataset, **kwargs):
    # if dataset == 'mnist':
    #     create_mnist_label_dir(get_report_dir())
    # total_count = 0
    # adv_exps = []
    # labels = []
    # for _, (x, y) in tqdm(enumerate(loader), total=len(loader)):
    #     x_prime, y = PGD(x, y, model, **kwargs)
    #    for j in range(adv_exp.shape[0]):
    #        save_image(adv_exp[j, :, :, :], f'{get_report_dir()}/{y[j]}/{total_count}.png')
    #        total_count += 1
    #     adv_exps.append(x_prime)
    #     labels.append(y)

    # print(len(adv_exps))
    # print(len(labels))
    # adv_exps = torch.stack(adv_exps[:-1])
    # labels = torch.stack(labels[:-1])
    # return adv_exps, labels

           

def get_PGD_kwargs(args):
    kwargs = {
        'loss': get_criterion(args.loss),
        'niter': args.niter,
        'epsilon': args.epsilon,
        'stepsize': args.stepsize,
        'randint': args.randint,
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
    parser.add_argument('--randint', action='store_true',
                        help='specify if start at random location for PGD (default= False)')  

    args = parser.parse_args()
    print(args)
    set_seed(113)

    model, input = get_model(args, 10)
    transform = get_transform(args, input)
    criterion = get_criterion(args.loss)
    test_loader = data_loader(args.dataset, 32, transform, train=False)

    model.eval()
    sum_loss = 0
    correct = 0
    total = 0
    for i, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        data, target = data.to(device), target.to(device)
        data, _ = PGD(data, target, model, **get_PGD_kwargs(args))
        pred = model(data)
        _, pred_label = torch.max(pred, 1)

        loss = criterion(pred, target)

        sum_loss += loss.item()
        correct += (pred_label == target).sum().item()
        total += len(data)

    print(f'loss: {sum_loss/total}, acc: {correct/total}')
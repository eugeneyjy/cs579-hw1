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


def craft_adv_exp(model, loader, dataset, **kwargs):
    if dataset == 'mnist':
        create_mnist_label_dir(get_report_dir())
    total_count = 0
    for _, (x, y) in tqdm(enumerate(loader), total=len(loader)):
       adv_exp, _ = PGD(x, y, model, **kwargs)
       for j in range(adv_exp.shape[0]):
           save_image(adv_exp[j, :, :, :], f'{get_report_dir()}/{y[j]}/{total_count}.png')
           total_count += 1
           

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
    test_loader = data_loader(args.dataset, 32, transform, train=False)
    craft_adv_exp(model, test_loader, args.dataset, **get_PGD_kwargs(args))
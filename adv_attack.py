import argparse
import torch 

from helper import get_model, get_criterion, get_transform, set_seed, show_image, map_label, device
from datasets import data_loader
from torchvision import transforms


def PGD(x, y, model, loss, niter=5, epsilon=0.03, stepsize=2/255, randint=True):
    show_image(x[0], map_label(y[0]))
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
        show_image(adv_imgs[0].cpu(), map_label(y[0]))
    
    return adv_imgs


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
    set_seed(113)

    model, input = get_model(args, 10)
    transform = get_transform(args, input)
    criterion = get_criterion(args.loss)
    test_loader = data_loader(args.dataset, 32, transform, train=False)

    image, label = next(iter(test_loader))
    PGD(image, label, model, criterion)
    # image.requires_grad = True
    # print(image.grad)
    # show_image(image[0], label[0].item())
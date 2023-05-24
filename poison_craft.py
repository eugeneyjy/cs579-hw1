import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from path import DATASETS_DIR

from helper import get_model, get_criterion, get_optimizer, get_transform, get_plot_title, set_seed, show_image, map_label, device

def craft_random_lflip(train_set, ratio):
    total_size = len(train_set)
    rand_inds = np.random.choice(total_size, int(total_size*ratio), replace=False)
    for ind in rand_inds:
        label = train_set.dataset.targets[train_set.indices[ind]]
        train_set.dataset.targets[train_set.indices[ind]] = 1 - label

    # # only flip label of all the 1s
    # changed = 0
    # i = len(train_set.indices)-1
    # while changed < int(total_size*ratio):
    #     label = train_set.dataset.targets[train_set.indices[i-changed]]
    #     if label == 0:
    #         train_set.dataset.targets[train_set.indices[i-changed]] = 1 - label
    #         changed += 1
    #     else:
    #         i -= 1

    return train_set

def craft_clabel_poisons(model, target, bases, niter=200, lr=0.01, beta=0.25):
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    all_poisons = []
    target_feature = feature_extractor(target)

    for i, (base_img, label) in tqdm(enumerate(bases), total=len(bases)):
        base_img = base_img.to(device)
        curr_img = base_img.clone().to(device)
        curr_img.requires_grad = True
        # print(f'ori: {torch.norm(feature_extractor(curr_img) - target_feature)}')

        for _ in range(niter):
            base_feature = feature_extractor(curr_img)

            # forward
            distance = torch.norm(base_feature - target_feature)
            grad = torch.autograd.grad(distance, curr_img)[0]
            curr_img = curr_img - (lr*grad)

            # backward
            curr_img = (curr_img + lr*beta*base_img) / (1 + beta*lr)
            curr_img = torch.clip(curr_img, 0, 1)
        # print(f'after: {torch.norm(feature_extractor(curr_img) - target_feature)}')

        curr_img = curr_img.squeeze()
        all_poisons.append(curr_img.cpu().detach().numpy())
    return np.array(all_poisons)


# function to get random image from a certain class
def load_cifar10_test(class_num, size, transform=None):
    data = datasets.CIFAR10(root=DATASETS_DIR, train=False, transform=transform, download=True)
    data.targets = np.array(data.targets)
    data.data = np.array(data.data)

    class_inds = (data.targets==class_num)
    data.targets = data.targets[class_inds]
    data.data = data.data[class_inds]

    rand_inds = np.random.choice(len(data.data), size, replace=False)
    print(rand_inds)
    data.targets = data.targets[rand_inds]
    data.data = data.data[rand_inds]

    loader = DataLoader(
        dataset=data,
        shuffle=False
    )

    return loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='lenet', metavar='lenet',
                        help='model architecture to train on (default: lenet)')
    parser.add_argument('--path', type=str, help='path that the model get save on')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='mnist',
                        help='dataset to evaluate on (default: mnist)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='64',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--loss', type=str, default='crossEntropy', metavar='crossEntropy',
                        help='loss function (default= crossEntropy)')
    parser.add_argument('--seed', type=int, default=113, metavar='113',
                        help='specify seed for random (default: 113)')
    
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    model, input_size = get_model(args, 10)

    craft_transform = get_transform(args, input_size)

    poison_dir = f'{DATASETS_DIR}/cifar-10-poison'
    os.makedirs(poison_dir, exist_ok=True)

    # Choose 5 frogs and 100 dogs images
    frog_loader = load_cifar10_test(6, 5, craft_transform)
    dog_loader = load_cifar10_test(5, 100, craft_transform)

    for i, (image, label) in enumerate(frog_loader):
        image = image.to(device)
        poisons = craft_clabel_poisons(model, image, dog_loader)
        np.save(f'{poison_dir}/poisons-{i}.npy', poisons)
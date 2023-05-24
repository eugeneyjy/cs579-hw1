#!/bin/bash

python adv_train.py --arch=resnet18 --batch-size=128 --epochs=30 --lr=0.0001 --dataset=cifar10 --optimizer=adam --epsilon=0.03 --randint

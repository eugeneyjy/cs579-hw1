#!/bin/bash

# python poison.py --arch=resnet18 --path=./models/pretrained/resnet-cifar10-ori/best.pth --batch-size=128 --epochs=10 --lr=0.001 --dataset=cifar10 --optimizer=adam --poison-ratio=$1 --poison-setId=$2
python poison.py --arch=logistic --batch-size=128 --epochs=10 --lr=0.01 --dataset=mnist --optimizer=adam --poison-ratio=$1 --subsample 1 7

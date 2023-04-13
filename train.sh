#!/bin/bash

python train.py --arch=lenet --batch-size=128 --epochs=30 --lr=0.0005 --dataset=mnist --optimizer=adam --no-save
#!/bin/bash

python adv_train.py --arch=lenet --batch-size=128 --epochs=30 --lr=0.001 --dataset=mnist --optimizer=adam --randint
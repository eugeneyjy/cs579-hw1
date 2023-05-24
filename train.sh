#!/bin/bash

python train.py --arch=logistic --batch-size=128 --epochs=30 --lr=0.001 --dataset=mnist --optimizer=adam --subsample 1 7

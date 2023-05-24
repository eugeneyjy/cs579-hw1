#!/bin/bash
poisonSetId=(0 1 2 3 4)
targetId=(641 987 187 779 848)
poisonSize=(1 5 10 25 50 100)
modelDir="/nfs/hpc/share/yonge/model"

# # For Random Flip
# poisonRatio=(0.00 0.05 0.10 0.25 0.50)
# for ratio in ${poisonRatio[@]}; do
#     echo "python valid.py --arch logistic --dataset mnist --path ${modelDir}/LogisticRegression_${ratio}/last.pth --subsample 1 7"
#     python valid.py --arch logistic --dataset mnist --path ${modelDir}/LogisticRegression_${ratio}/last.pth --subsample 1 7
# done

# For Poison Frog
for index in ${!poisonSetId[@]}; do
    echo "python valid.py --arch resnet18 --dataset cifar10 --path ${modelDir}/pretrained/resnet-cifar10-ori/best.pth --targetId ${targetId[$index]}"
    python valid.py --arch resnet18 --dataset cifar10 --path ${modelDir}/pretrained/resnet-cifar10-ori/best.pth --targetId ${targetId[$index]}
    for size in ${poisonSize[@]}; do
        echo "python valid.py --arch resnet18 --dataset cifar10 --path ${modelDir}/ResNet18_T_${poisonSetId[$index]}_P_${size}/last.pth --targetId ${targetId[$index]}"
        python valid.py --arch resnet18 --dataset cifar10 --path ${modelDir}/ResNet18_T_${poisonSetId[$index]}_P_${size}/last.pth --targetId ${targetId[$index]}
    done
done
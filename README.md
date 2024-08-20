# Multi-consistency for Semi-Supervised medical image segmentation with Diffusion Models

## Introduction
This repository is the Pytorch implementation of "Multi-consistency for Semi-Supervised medical image segmentation with Diffusion Models"

## Requirements
We implemented our experiment on the super-parallel computer system of Guangxi University. The specific configuration is as follows:
* Centos 7.4
* NVIDIA Tesla V100 32G
* Intel Xeon gold 6230 2.1G 20C processor


# Datasets
ACDC
https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC
M&Ms
https://www.ub.edu/mnms

# Usage

1. Clone the repo:
```
git clone git@github.com:yunzhuC/MCSD.git
cd MCSD
```

2.. Train the model
```
python train.py
```

4. Test the model
```
python test.py
```

## Citation

## Acknowledgement
Part of the code is revised from the [UniMatch](https://github.com/LiheYoung/UniMatch/tree/main).

## Note
* The repository is being updated.

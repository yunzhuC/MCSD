# Multi-consistency for Semi-Supervised medical image segmentation with Diffusion Models

## Introduction
This repository is the Pytorch implementation of "Multi-consistency for Semi-Supervised medical image segmentation with Diffusion Models"

## Requirements
We implemented our experiment on the super-parallel computer system of Guangxi University. The specific configuration is as follows:
* Centos 7.4
* NVIDIA Tesla V100 32G
* Intel Xeon gold 6230 2.1G 20C processor


## Dataset
Please modify your dataset path.
* ACDC: 
https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html |
https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC
* M&Ms: 
https://www.ub.edu/mnms

## Comparison of network complexity with other methods

Our mode shows a significant improvement in segmentation performance while keeping an acceptable increase of parameter number and a competitive per-image inference time against other SOTA methods. Specifically, we offer a feature dimensionality reduction optimization strategy to control the growth in the number of parameters while also improving the model's training and inference efficiency. This method not only effectively minimizes the number of parameters, but also assures that our model's prediction time per image does not increase much when compared to other models. We intend to optimize the model structure further to reduce the number of parameters while maintaining or improving segmentation performance..


Method	Parameters (M)	Prediction time 
per image (s)
UA-MT [23]	1.81	0.48
URPC [37]	1.81	0.48
CPS [38]	1.81	0.48 
CNN & Trans [33]	1.81	0.48
MC-NET+ [26]	2.58	0.53 
BCP [28]	1.81	0.65 
Ours	3.63	0.55 



| Method                      | Parameters (M) | Prediction time per image (s) | 1/4 (366) | 1/2 (732) | Full (1464) |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: |
| UA-MT                 | 1.81      | 0.48      | | | |
| URPC             | 1.81      | 0.48      | | | |
| CPS             | 1.81     | 0.48     |       | | |
| CNN & Trans             | 1.81| 0.48|       | | |
| MC-NET+             | 2.58      | 0.53      |       | | |
| BCP                        | 1.81      | 0.65      | |       | |
| **Ours**         | **3.63**  | **0.55**  |    



## Usage

1. Clone the repo:
```
git clone git@github.com:yunzhuC/MCSD.git
cd MCSD
```

2. Train the model
```
python train.py
```

3. Test the model
```
python test.py
```

## Citation

## Acknowledgement
Part of the code is revised from the [UniMatch](https://github.com/LiheYoung/UniMatch/tree/main).

## Note
* The repository is being updated.

# Comparisong of SIFT and CNN

This project compares using SIFT with color features and both trained and pretrained CNNs in a butterfly classification task.

The `data_pipeline` folder contains the different datasets used, implemented as PyTorch datasets and dataloaders.
The `models` folder contains the baseline CNN used in the project.
The `training` has all the code required to train the baseline CNN and the fine-tuned ImageNet classifier.
The `classifier` has the scripts used to obtain results from trained CNNs and SIFT features with an SVM.

## The butterfly dataset

This Butterfly-200 dataset used in this project is not included in the repo. You can download it from [here](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0), and move it to a folder called `data` in the root.
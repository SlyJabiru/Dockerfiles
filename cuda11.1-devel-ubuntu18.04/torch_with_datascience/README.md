# Torch With Datascience

## Requirement
* docker
* docker-compose
* nvidia-docker (to use GPU)

## Usage
```
sudo docker build --tag ${your_image_name}:${tag} -f Dockerfile .
sudo docker-compose -f docker-compose.yml up
```
And then run multi_gpu_examples.ipynb !

## config
* Base = nvidia/cuda:11.1-devel-ubuntu18.04
* Target GPU = RTX 3090
* Python = 3.8
* Base Dependence Manager = Conda

## Arguments
* PYTHON_VERSION=3.8
* ARG WITH_TORCHVISION=1
* ARG CUDATOOLKIT_VERSION=11.0
* ARG CMAKE_MAJOR_VERSION=3.19
* ARG CMAKE_MINOR_VERSION=1
* ARG CMAKE_FULL_VERSION=3.19.1

## Installed Main Python Packages
* pytorch
* scikit-learn
* pandas
* seaborn
* numpy
* matplotlib
* hyperopt
* shap
* xgboost with gpu support
* lightgbm with gpu support
* jupyter lab

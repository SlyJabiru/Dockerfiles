# Torch With Datascience

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

### For now, I can't run multi-gpu pytorch training with this dockerfile
### More environment variable should be handled
## However, I can use single GPU properly now with this dockerfile!


#### Multi GPU Training Reference: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
FROM nvidia/cuda:11.1-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8
ARG WITH_TORCHVISION=1


# Install necessary building tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         sudo \
         bzip2 \
         libx11-6 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y dist-upgrade && apt-get purge -y libboost-all-dev && \
    apt-get install -f -y libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /app
WORKDIR /app


# Install conda and torch
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
    /opt/conda/bin/conda clean -ya

ARG CUDATOOLKIT_VERSION=11.0
ENV CONDA_OVERRIDE_CUDA=11.0
RUN /opt/conda/bin/conda install -y pytorch torchvision torchaudio cudatoolkit=$CUDATOOLKIT_VERSION -c pytorch && \
    /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/pip install torchelastic


ENV PATH /opt/conda/bin:$PATH

RUN apt-get update
RUN apt-get install -y vim lshw

ENV LANG C.UTF-8


# Install data science and machine learning packages using conda
RUN /opt/conda/bin/conda install -y numpy scikit-learn pandas seaborn && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y -c conda-forge matplotlib hyperopt shap && \
    /opt/conda/bin/conda install -c anaconda python-dateutil && \
    /opt/conda/bin/conda clean -ya

RUN apt install -y python-pydot python-pydot-ng graphviz


# Install cmake
RUN apt-get install wget libssl-dev

ARG CMAKE_MAJOR_VERSION=3.19
ARG CMAKE_MINOR_VERSION=1
ARG CMAKE_FULL_VERSION=$CMAKE_MAJOR_VERSION.$CMAKE_MINOR_VERSION
RUN cd ~ && \
    wget http://cmake.org/files/v$CMAKE_MAJOR_VERSION/cmake-$CMAKE_FULL_VERSION.tar.gz && \
    tar xf cmake-$CMAKE_FULL_VERSION.tar.gz && \
    cd ~/cmake-$CMAKE_FULL_VERSION && \
    ./configure && \
    make && \
    make install
ENV PATH /usr/local/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH


# Install XGBoost GPU version
RUN cd ~ && \
    git clone --recursive https://github.com/dmlc/xgboost

RUN cd ~/xgboost && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON && \
    make -j4

RUN cd ~/xgboost/python-package && \
    apt-get install -y python-setuptools && \
    python3 setup.py install && \
    python3 setup.py develop --user
ENV PYTHONPATH ~/xgboost/python-package:$PYTHONPATH


# Install LightGBM GPU version
RUN cd ~ && \
    git clone --recursive https://github.com/microsoft/LightGBM

RUN cd ~/LightGBM && \
    mkdir build && \
    cd build && \
    cmake -DUSE_GPU=1 .. && \
    make -j4

RUN cd ~/LightGBM/python-package && \
    python3 setup.py install --precompile
ENV PYTHONPATH ~/LightGBM/python-package:$PYTHONPATH


# Install Jupyter
RUN /opt/conda/bin/conda install -y -c conda-forge jupyterlab nodejs  && \
    /opt/conda/bin/conda install -y -c conda-forge  jupyter_http_over_ws  && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y -c anaconda nbformat && \
    /opt/conda/bin/conda install -y ipykernel && \
    /opt/conda/bin/conda clean -ya
RUN jupyter serverextension enable --py jupyter_http_over_ws


EXPOSE 9989

RUN python3 -m ipykernel.kernelspec

# CMD ["bash", "-c", "jupyter lab --notebook-dir=/path/to/dir --port=9989 --no-browser --allow-root"]
# sudo docker run -it -p 9989:9989 onmybeamer_torch:0.6 /bin/bash

# You should do port forwarding to see jupyter of docker container of remote server in your local machine!
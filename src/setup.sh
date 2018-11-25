# #!/bin/bash
# 
# # Set up a new GCP instance with everything needed to run a ULMFiT model
set -x
# 
# # Don't download stuff to the git repo, that's messy.
pushd ${HOME}
# 
# Update packages
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install bzip2

ANACONDA_INSTALLER="Anaconda3-5.2.0-Linux-x86_64.sh"
wget "https://repo.continuum.io/archive/$ANACONDA_INSTALLER"
bash "$ANACONDA_INSTALLER"

source ${HOME}/.bashrc

TF_BINARY_URL="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp36-cp36m-linux_x86_64.whl"
${HOME}/anaconda3/bin/pip install --upgrade pip
${HOME}/anaconda3/bin/pip install $TF_BINARY_URL
${HOME}/anaconda3/bin/jupyter notebook --generate-config

# Copy Jupyter config
popd
mkdir ${HOME}/.jupyter
cp -v $(dirname $0)/support/jupyter_notebook_config.py ${HOME}/.jupyter/jupyter_notebook_config.py

export PATH=~/anaconda3/bin:$PATH

conda install nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager

conda install -c pytorch pytorch-nightly cuda92
conda install -c fastai torchvision-nightly
conda install -c fastai fastai

sudo apt-get install graphviz

conda install -c anaconda graphviz
conda install -c conda-forge python-graphviz
conda install pydot

sudo apt-get install xdg-utils

conda install networkx

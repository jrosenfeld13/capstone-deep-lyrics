#!/bin/bash

##########################################################
# Server setup scripts for DeepLyrics
# Assumes server is a Ubuntu 16.04 image with a GPU (K80)
##########################################################

# explicity print commands
set -x

sudo apt-get update
sudo apt-get upgrade --assume-yes
sudo apt-get install bzip2 --assume-yes

ANACONDA_INSTALLER="Anaconda3-5.2.0-Linux-x86_64.sh"
wget "https://repo.continuum.io/archive/$ANACONDA_INSTALLER"
bash "$ANACONDA_INSTALLER" -b

echo export PATH=~/anaconda3/bin:$PATH >> ${HOME}/.bashrc
source ${HOME}/.bashrc

# set up conda channels
conda config --add channels pytorch
conda config --add channels fastai
conda config --append channels conda-forge

# NVIDIA Drivers
# https://websiteforstudents.com/install-proprietary-nvidia-gpu-drivers-on-ubuntu-16-04-17-10-18-04/
sudo add-apt-repository ppa:graphics-drivers/ppa --yes
sudo apt update
sudo apt install nvidia-387 --assume-yes

sudo apt-get install xdg-utils --assume-yes
sudo apt-get install graphviz --assume-yes

# installation outside of conda
pip install pronouncing
python -m nltk.downloader 'punkt'
python -m nltk.downloader 'averaged_perceptron_tagger'

# clone repo -- authentication is required
git clone https://github.com/jrosenfeld13/capstone-deep-lyrics.git

echo "To complete server setup run `conda install --yes --file conda_requirements.txt` from git repo."

#
# # Jupyter Lab
# To configure run `jupyter notebook --generate-config`
# Then update NotebookApp.ip setting to '*'
# conda install nodejs
# jupyter labextension install @jupyter-widgets/jupyterlab-manager

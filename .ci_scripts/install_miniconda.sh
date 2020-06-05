#!/bin/bash

echo "installing miniconda"
rm -rf $HOME/miniconda

mkdir -p $HOME/download

curl -s https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $HOME/download/miniconda.sh;

bash $HOME/download/miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH

cp .ci_scripts/condarc $HOME/miniconda/.condarc
conda update -q conda
conda info -a

if [ "${TOXENV}" = py36 ]; then
    pyver=3.6
fi
if [ "${TOXENV}" = py37 ]; then
    pyver=3.7
fi

conda create -q -n test-env python=${pyver} pip setuptools \
    numpy numba flake8 pyyaml scipy pytest galsim \
    scikit-learn statsmodels emcee

conda clean --all

[![Build Status](https://travis-ci.org/mackelab/pyknos.svg?branch=master)](https://travis-ci.org/mackelab/pyknos)



## Description
Building on code for "On Contrastive Learning for Likelihood-free Inference" in https://github.com/conormdurkan/lfi, the relevant part is mostly from https://github.com/bayesiains/nsf.

Features two families of neural conditional density estimators: normalizing flows and mixture-density networks. 

## Setup

You can install all the dependencies using the `environment.yml` file to create a conda environment: `conda env create -f environment.yml`

Alternatively, you can install via `setup.py` using `pip install -e.`

## Examples

Examples are collected in notebooks in `examples/`. 

## Git LFS

We use git lfs to store binary files, e.g., example notebooks. To use git lfs follow installation instructions here https://git-lfs.github.com/. 

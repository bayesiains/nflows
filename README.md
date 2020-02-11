[![Build Status](https://travis-ci.org/mackelab/sbi.svg?branch=master)](https://travis-ci.org/mackelab/sbi)



## Description
Building on code for "On Contrastive Learning for Likelihood-free Inference" in https://github.com/conormdurkan/lfi . 

Features neural likelihood-free methods from

> Papamakarios et al., _Sequential Neural Likelihood_ (SNL), 2019. [[arXiv]](https://arxiv.org/abs/1805.07226)
>Greenberg et al., _Automatic Posterior Transformation_ (SNPE-C), 2019. [[arXiv]](https://arxiv.org/abs/1905.07488)
>Hermans et al., _Likelihood-free Inference with Amortized Approximate Likelihood Ratios_ (SRE), 2019.  [[arXiv]](https://arxiv.org/abs/1903.04057)

## Setup

Clone repo, then set environment variable ```LFI_PROJECT_DIR``` to local directory.  

You can install all the dependencies using the `environment.yaml` file to create a conda environment: `conda env create -f environment.yaml' 

Alternatively, you can install via `setup.py` using `pip install -e.`

Also uses https://github.com/bayesiains/nsf for general density estimation, but that directory is included here so you don't need to get it separately (will hopefully be a pip installable package soon, and in PyTorch master some day).  

## Examples

Examples are collected in notebooks in `examples/`. 

## Git lfs

We use git lfs to store binary files, e.g., example notebooks. To use git lfs follow installation instructions here https://git-lfs.github.com/. 

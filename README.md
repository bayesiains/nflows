[![Build Status](https://travis-ci.org/mackelab/pyknos.svg?branch=master)](https://travis-ci.org/mackelab/pyknos)


## Description
Building on code for "On Contrastive Learning for Likelihood-free Inference" in https://github.com/conormdurkan/lfi, the relevant part is mostly from https://github.com/bayesiains/nsf.

Features two families of neural conditional density estimators: normalizing flows and mixture-density networks. 

## Setup

You can install all the dependencies using the `environment.yml` file to create a conda environment: `conda env create -f environment.yml`

Alternatively, you can install via `setup.py` using `pip install -e.` To install development and testing dependencies, if you use bash use `pip install -e .[dev]` while if you use zsh do `pip install -e .\[dev\]`.

## Examples

Examples are collected in notebooks in `examples/`. 

## Git LFS

We use git lfs to store binary files, e.g., example notebooks. To use git lfs follow installation instructions here https://git-lfs.github.com/. 

## Acknowledgements
This code builds heavily on previous work by [Conor Durkan](https://conormdurkan.github.io/), [George Papamakarios](https://gpapamak.github.io/) and [Artur Bekasov](https://arturbekasov.github.io/).
Relevant repositories include [bayesiains/nsf](https://github.com/bayesiains/nsf) and [conormdurkan/lfi](https://github.com/conormdurkan/lfi). 
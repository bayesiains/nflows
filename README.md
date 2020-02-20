## Description
`nflows` is a comprehensive collection of [normalizing flows](https://arxiv.org/abs/1912.02762) using [PyTorch](https://pytorch.org).

## Setup

You can install all the dependencies using the `environment.yml` file to create a conda environment: `conda env create -f environment.yml`

Alternatively, you can install via `setup.py` using `pip install -e ".[dev]"` (the `dev` flag installs development and testing dependencies).


## References
`nflows` is derived from [bayesiains/nsf](https://github.com/bayesiains/nsf) originally published with
> C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, _Neural Spline Flows_, NeurIPS 2019.
> [[arXiv]](https://arxiv.org/abs/1906.04032) [[bibtex]](https://gpapamak.github.io/bibtex/neural_spline_flows.bib)


`nflows` have been used as density estimators for likelihood-free inference in 
> Conor Durkan, Iain Murray, George Papamakarios, _On Contrastive Learning for Likelihood-free Inference_
> [[arXiv]](https://arxiv.org/abs/2002.03712).
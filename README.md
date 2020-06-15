## Description
`nflows` is a comprehensive collection of [normalizing flows](https://arxiv.org/abs/1912.02762) using [PyTorch](https://pytorch.org).

## Installation

To install from PyPI:
```
pip install nflows
```

### For development

You can install all the dependencies using the `environment.yml` file to create a conda environment: `conda env create -f environment.yml`

Alternatively, you can install via `setup.py` using `pip install -e ".[dev]"` (the `dev` flag installs development and testing dependencies).

## Usage

![diagram.png](https://raw.githubusercontent.com/arashabzd/nflows/better-readme%233/diagram.png "Diagram")

To construct a flow you need to inherit from `Flow` class and provide its constructor with three parameters:

- `transform`: A `Transform` object that can be a composition of several invertible transformations from data to the base distribution. There are a plethora of transformations defined in `nflows.transformations` module (e.x. coupling, autoregressive, spline, ...).

- `distribution`: A `Distribution` object that specifies the base distribution of flow. Could be a conditional distribution in case any conditioning variable is available. There are several distributions defined in `nflows.distributions` module.

- `embedding_net` (Optional): An `nn.Module` object that encodes conditioning variable if available. Output of this network is the context that gets fed to the transform and distribution objects.

Additional examples of the workflow are provided in [examples folder](https://github.com/arashabzd/nflows/tree/better-readme%233/examples).

## References
`nflows` is derived from [bayesiains/nsf](https://github.com/bayesiains/nsf) originally published with
> C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, _Neural Spline Flows_, NeurIPS 2019.
> [[arXiv]](https://arxiv.org/abs/1906.04032) [[bibtex]](https://gpapamak.github.io/bibtex/neural_spline_flows.bib)


`nflows` have been used as density estimators for likelihood-free inference in 
> Conor Durkan, Iain Murray, George Papamakarios, _On Contrastive Learning for Likelihood-free Inference_
> [[arXiv]](https://arxiv.org/abs/2002.03712).

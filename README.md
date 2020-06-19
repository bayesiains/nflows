## Description
`nflows` is a comprehensive collection of [normalizing flows](https://arxiv.org/abs/1912.02762) using [PyTorch](https://pytorch.org).

## Installation

To install from PyPI:
```
pip install nflows
```

## Usage

To define a flow:

```python
from nflows import transforms, distributions, flows

# Define an invertible transformation.
transform = transforms.CompositeTranform([
    transforms.AffineCouplingTransform(...),
    transforms.RandomPermutation(...)
])

# Define a base distribution.
base_distribution = distributions.StandardNormal(...)

# Combine into a flow.
flow = flows.Flow(transform=transform, distribution=base_distribution)
```

To evaluate log probabilities of inputs:
```python
log_prob = flow.log_prob(inputs)
```

To sample from the flow:
```python
samples = flow.sample(num_samples)
```

Additional examples of the workflow are provided in [examples folder](examples/).

## Development

You can install all the dependencies using the `environment.yml` file to create a conda environment: 
```
conda env create -f environment.yml
```

Alternatively, you can install via `setup.py` (the `dev` flag installs development and testing dependencies):
```
pip install -e ".[dev]"
```

## References
`nflows` is derived from [bayesiains/nsf](https://github.com/bayesiains/nsf) originally published with
> C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, _Neural Spline Flows_, NeurIPS 2019.
> [[arXiv]](https://arxiv.org/abs/1906.04032) [[bibtex]](https://gpapamak.github.io/bibtex/neural_spline_flows.bib)


`nflows` have been used as density estimators for likelihood-free inference in 
> Conor Durkan, Iain Murray, George Papamakarios, _On Contrastive Learning for Likelihood-free Inference_
> [[arXiv]](https://arxiv.org/abs/2002.03712).

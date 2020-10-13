# nflows

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
transform = transforms.CompositeTransform([
    transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=4),
    transforms.RandomPermutation(features=2)
])

# Define a base distribution.
base_distribution = distributions.StandardNormal(shape=[2])


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


`nflows` has been used in 
> Conor Durkan, Iain Murray, George Papamakarios, _On Contrastive Learning for Likelihood-free Inference_, ICML 2020.
> [[arXiv]](https://arxiv.org/abs/2002.03712).

> Artur Bekasov, Iain Murray, _Ordering Dimensions with Nested Dropout Normalizing Flows_.
> [[arXiv]](https://arxiv.org/abs/2006.08777).

> Tim Dockhorn, James A. Ritchie, Yaoliang Yu, Iain Murray, _Density Deconvolution with Normalizing Flows_.
> [[arXiv]](https://arxiv.org/abs/2006.09396).

`nflows` is used by the conditional density estimation package [pyknos](https://github.com/mackelab/pyknos), and in turn the likelihood-free inference framework [sbi](https://github.com/mackelab/sbi).

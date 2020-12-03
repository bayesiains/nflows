# nflows

<a href="https://doi.org/10.5281/zenodo.4296287"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4296287.svg" alt="DOI"></a>

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

## Citing nflows

To cite the package:
```bibtex
@software{nflows,
  author       = {Conor Durkan and
                  Artur Bekasov and
                  Iain Murray and
                  George Papamakarios},
  title        = {{nflows}: normalizing flows in {PyTorch}},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.14},
  doi          = {10.5281/zenodo.4296287},
  url          = {https://doi.org/10.5281/zenodo.4296287}
}
```

The version number is intended to be the one from `nflows/version.py`. The year/month correspond to the date of the release. BibTeX entries for other versions could be found on [Zenodo](https://doi.org/10.5281/zenodo.4296286).

If you're using spline-based flows in particular, consider citing the _Neural Spline Flows_ paper: [[bibtex]](https://papers.nips.cc/paper/2019/file/7ac71d433f282034e088473244df8c02-Bibtex.bib).

## References
`nflows` is derived from [bayesiains/nsf](https://github.com/bayesiains/nsf) originally published with
> C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, _Neural Spline Flows_, NeurIPS 2019.
> [[arXiv]](https://arxiv.org/abs/1906.04032) [[bibtex]](https://papers.nips.cc/paper/2019/file/7ac71d433f282034e088473244df8c02-Bibtex.bib)


`nflows` has been used in 
> Conor Durkan, Iain Murray, George Papamakarios, _On Contrastive Learning for Likelihood-free Inference_, ICML 2020.
> [[arXiv]](https://arxiv.org/abs/2002.03712).

> Artur Bekasov, Iain Murray, _Ordering Dimensions with Nested Dropout Normalizing Flows_.
> [[arXiv]](https://arxiv.org/abs/2006.08777).

> Tim Dockhorn, James A. Ritchie, Yaoliang Yu, Iain Murray, _Density Deconvolution with Normalizing Flows_.
> [[arXiv]](https://arxiv.org/abs/2006.09396).

`nflows` is used by the conditional density estimation package [pyknos](https://github.com/mackelab/pyknos), and in turn the likelihood-free inference framework [sbi](https://github.com/mackelab/sbi).

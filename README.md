## Description
Code for "On Contrastive Learning for Likelihood-free Inference".

Features neural likelihood-free methods from

> Papamakarios et al., _Sequential Neural Likelihood_ (SNL), 2019. [[arXiv]](https://arxiv.org/abs/1805.07226)

>Greenberg et al., _Automatic Posterior Transformation_ (SNPE-C), 2019. [[arXiv]](https://arxiv.org/abs/1905.07488)

>Hermans et al., _Likelihood-free Inference with Amortized Approximate Likelihood Ratios_ (SRE), 2019.  [[arXiv]](https://arxiv.org/abs/1903.04057)

## Setup

Clone repo, then set environment variable ```LFI_PROJECT_DIR``` to local directory.  

Required packages: NumPy, SciPy, Matplotlib, tqdm, PyTorch, Pyro, and TensorBoard.

Also uses https://github.com/bayesiains/nsf for general density estimation, but that directory is included here so you don't need to get it separately (will hopefully be a pip installable package soon, and in PyTorch master some day).  

## Examples
Each inference method logs TensorBoard output in ```$(LFI_PROJECT_DIR)/log``` for each run.  

#### SNPE-C 
To use SNPE-C with Nonlinear Gaussian simulator:
```python
import torch

import inference
import simulators
import utils

from matplotlib import pyplot as plt

# use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

# get simulator and prior
simulator, prior = simulators.get_simulator_and_prior("nonlinear-gaussian")

# get neural posterior (here a MAF)
neural_posterior = utils.get_neural_posterior(
    "maf",
    parameter_dim=simulator.parameter_dim,
    observation_dim=simulator.observation_dim,
    simulator=simulator,
)

# create inference method
inference_method = inference.APT(
    simulator=simulator,
    prior=prior,
    true_observation=simulator.get_ground_truth_observation(),
    neural_posterior=neural_posterior,
    num_atoms=-1,
)

# run inference
inference_method.run_inference(num_rounds=10, num_simulations_per_round=1000)

# sample posterior
samples = inference_method.sample_posterior(num_samples=10000)

# plot samples
utils.plot_hist_marginals(
    utils.tensor2numpy(samples),
    lims=simulator.parameter_plotting_limits,
    ground_truth=utils.tensor2numpy(simulator.get_ground_truth_parameters()).reshape(
        -1
    ),
)
plt.show()
```
Note that SNPE-C does density estimation on parameters rather than observations as in SNL, so we need to normalize parameters to a reasonable scale to make things sensible (really only needed for M/G/1 and Lotka-Volterra because of wide box uniform priors). This is handled in the example above by the ```get_neural_posterior``` method, but in general you'll have to make sure your density estimator is receiving parameters on a reasonable scale, and also that the change in density caused by normalization is accounted for. 

#### SNL
To use SNL with M/G/1 simulator:
```python
import torch

import inference
import simulators
import utils

from matplotlib import pyplot as plt

# use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

# get simulator and prior
simulator, prior = simulators.get_simulator_and_prior("mg1")

# get neural likelihood
neural_likelihood = utils.get_neural_likelihood(
    "maf",
    parameter_dim=simulator.parameter_dim,
    observation_dim=simulator.observation_dim,
)

# create inference method
inference_method = inference.SNL(
    simulator=simulator,
    prior=prior,
    true_observation=simulator.get_ground_truth_observation(),
    neural_likelihood=neural_likelihood,
    mcmc_method="slice-np",
)

# run inference
inference_method.run_inference(num_rounds=10, num_simulations_per_round=1000)

# sample posterior
samples = inference_method.sample_posterior(num_samples=10000)

# plot samples
utils.plot_hist_marginals(
    utils.tensor2numpy(samples),
    lims=simulator.parameter_plotting_limits,
    ground_truth=utils.tensor2numpy(simulator.get_ground_truth_parameters()).reshape(
        -1
    ),
)
plt.show()
```

#### SRE
To use SRE with the Lotka-Volterra simulator: 
```python
import torch

import inference
import simulators
import utils

from matplotlib import pyplot as plt

# use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

# get simulator and prior
simulator, prior = simulators.get_simulator_and_prior("lotka-volterra")

# get classifier
classifier = utils.get_classifier(
    "resnet",
    parameter_dim=simulator.parameter_dim,
    observation_dim=simulator.observation_dim,
)

# create inference method
inference_method = inference.SRE(
    simulator=simulator,
    prior=prior,
    true_observation=simulator.get_ground_truth_observation(),
    classifier=classifier,
    mcmc_method="slice-np",
)

# run inference
inference_method.run_inference(num_rounds=10, num_simulations_per_round=1000)

# sample posterior
samples = inference_method.sample_posterior(num_samples=10000)

# plot samples
utils.plot_hist_marginals(
    utils.tensor2numpy(samples),
    lims=simulator.parameter_plotting_limits,
    ground_truth=utils.tensor2numpy(simulator.get_ground_truth_parameters()).reshape(
        -1
    ),
)
plt.show()
```
 

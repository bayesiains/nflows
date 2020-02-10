import torch

import lfi.utils as utils

from matplotlib import pyplot as plt
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from torch import distributions


class Slice(MCMCKernel):
    def __init__(self, potential_function, max_width=float("inf")):
        self.potential_function = potential_function
        self._max_width = max_width
        self._width = None
        self._current_parameters = None
        super().__init__()

    def setup(self, warmup_steps, *args, **kwargs):
        self._current_parameters = next(iter(self.initial_params.values())).clone()
        if self._width is None:
            self._tune_bracket_width()

    @property
    def features(self):
        return next(iter(self.initial_params.values())).shape[0]

    @property
    def initial_params(self):
        """
        Returns a dict of initial params (by default, from the prior) to initiate the MCMC run.

        :return: dict of parameter values keyed by their name.
        """
        return self._initial_parameters

    @initial_params.setter
    def initial_params(self, parameters):
        """
        Sets the parameters to initiate the MCMC run. Note that the parameters must
        have unconstrained support.
        """
        assert (
            isinstance(parameters, dict) and len(parameters) == 1
        ), "Slice sampling only implemented for a single site."
        self._initial_parameters = parameters

    def sample(self, parameters):
        """
        Samples parameters from the posterior distribution, when given existing parameters.

        :param dict params: Current parameter values.
        :param int time_step: Current time step.
        :return: New parameters from the posterior distribution.
        # """
        order = torch.randperm(self.features)
        site_name, self._current_parameters = next(iter(parameters.items()))
        for dim in order:
            self._current_parameters[dim], _ = self._sample_from_conditional(
                dim, self._current_parameters[dim]
            )
        return {site_name: self._current_parameters}.copy()

    def _tune_bracket_width(self):
        num_tuning_samples = 50
        order = torch.arange(self.features)
        parameters = next(iter(self.initial_params.values())).clone()
        self._width = torch.full((self.features,), 0.01)

        for n in range(num_tuning_samples):

            order = order[torch.randperm(self.features)]

            for dim in order:
                parameters[dim], width_d = self._sample_from_conditional(
                    dim, parameters[dim]
                )
                self._width[dim] += (width_d.item() - self._width[dim]) / (n + 1)

    def _sample_from_conditional(self, dim, parameter):

        # conditional log_prob
        a, b = self._current_parameters[:dim], self._current_parameters[dim + 1 :]
        log_prob_d = lambda x: -self.potential_function(
            {"": torch.cat((a, x.reshape(-1), b)).reshape(1, -1)}
        )
        bracket_width = self._width[dim]

        # sample a slice uniformly
        log_height = log_prob_d(parameter) + torch.log(torch.rand(1))

        # position the bracket randomly around the current sample
        lower = parameter - bracket_width * torch.rand(1)
        upper = lower + bracket_width

        # find lower bracket end
        while log_prob_d(lower) >= log_height and parameter - lower < self._max_width:
            lower -= bracket_width

        # find upper bracket end
        while log_prob_d(upper) >= log_height and upper - parameter < self._max_width:
            upper += bracket_width

        # sample uniformly from bracket
        new_parameter = (upper - lower) * torch.rand(1) + lower

        # if outside slice, reject sample and shrink bracket
        while log_prob_d(new_parameter) < log_height:
            if new_parameter < parameter:
                lower = new_parameter
            else:
                upper = new_parameter
            new_parameter = (upper - lower) * torch.rand(1) + lower

        return new_parameter, upper - lower

    def __call__(self, params):
        """
        Alias for MCMCKernel.sample() method.
        """
        return self.sample(params)


class PotentialFunction:
    def __init__(self, likelihood, prior):
        self.likelihood = likelihood
        self.prior = prior

    def __call__(self, parameters_dict):
        parameters = next(iter(parameters_dict.values()))
        return -(
            self.likelihood.log_prob(parameters) + self.prior.log_prob(parameters).sum()
        )


def test_():
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # else:
    #     input("CUDA not available, do you wish to continue?")
    #     device = torch.device("cpu")
    #     torch.set_default_tensor_type("torch.FloatTensor")

    loc = torch.Tensor([0, 0])
    covariance_matrix = torch.Tensor([[1, 0.99], [0.99, 1]])

    likelihood = distributions.MultivariateNormal(
        loc=loc, covariance_matrix=covariance_matrix
    )
    bound = 1.5
    low, high = -bound * torch.ones(2), bound * torch.ones(2)
    prior = distributions.Uniform(low=low, high=high)

    # def potential_function(inputs_dict):
    #     parameters = next(iter(inputs_dict.values()))
    #     return -(likelihood.log_prob(parameters) + prior.log_prob(parameters).sum())
    prior = distributions.Uniform(low=-5 * torch.ones(4), high=2 * torch.ones(4))
    from nsf import distributions as distributions_

    likelihood = distributions_.LotkaVolterraOscillating()
    potential_function = PotentialFunction(likelihood, prior)

    # kernel = Slice(potential_function=potential_function)
    from pyro.infer.mcmc import HMC, NUTS

    # kernel = HMC(potential_fn=potential_function)
    kernel = NUTS(potential_fn=potential_function)
    num_chains = 3
    sampler = MCMC(
        kernel=kernel,
        num_samples=10000 // num_chains,
        warmup_steps=200,
        initial_params={"": torch.zeros(num_chains, 4)},
        num_chains=num_chains,
    )
    sampler.run()
    samples = next(iter(sampler.get_samples().values()))

    utils.plot_hist_marginals(
        utils.tensor2numpy(samples), ground_truth=utils.tensor2numpy(loc), lims=[-6, 3]
    )
    # plt.show()
    plt.savefig("/home/conor/Dropbox/phd/projects/lfi/out/mcmc.pdf")
    plt.close()


def main():
    test_()


if __name__ == "__main__":
    main()

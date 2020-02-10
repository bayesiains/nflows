import numpy as np
import scipy.stats
import os
import torch

import lfi.utils as utils

from matplotlib import pyplot as plt

from lfi.mcmc import SliceSampler
from .simulator import Simulator

parameter_dim = 5
observation_dim = 8


class NonlinearGaussianSimulator(Simulator):
    """
    Implemenation of nonlinear Gaussian simulator as described in section 5.2 and appendix
    A.1 of 'Sequential Neural Likelihood'.
    """

    def __init__(self):
        super().__init__()
        self._num_observations_per_parameter = 4
        self._posterior_samples = None

    def simulate(self, parameters):
        """
        Generates observations for the given batch of parameters.

        :param parameters: torch.Tensor
            Batch of parameters.
        :return: torch.Tensor
            Batch of observations.
        """

        # Run simulator in NumPy.
        if isinstance(parameters, torch.Tensor):
            parameters = utils.tensor2numpy(parameters)

        # If we have a single parameter then view it as a batch of one.
        if parameters.ndim == 1:
            return self.simulate(parameters[np.newaxis, :])[0]

        num_simulations = parameters.shape[0]

        # Keep track of total simulations.
        self.num_total_simulations += num_simulations

        # Run simulator to generate self._num_observations_per_parameter
        # observations from a 2D Gaussian parameterized by the 5 given parameters.
        m0, m1, s0, s1, r = self._unpack_params(parameters)

        us = np.random.randn(num_simulations, self._num_observations_per_parameter, 2)
        observations = np.empty_like(us)

        observations[:, :, 0] = s0 * us[:, :, 0] + m0
        observations[:, :, 1] = (
            s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r ** 2) * us[:, :, 1]) + m1
        )

        mean, std = self._get_observation_normalization_parameters()
        return (
            torch.Tensor(
                observations.reshape(
                    [num_simulations, 2 * self._num_observations_per_parameter]
                )
            )
            - mean.reshape(1, -1)
        ) / std.reshape(1, -1)

    def log_prob(self, observations, parameters):
        """
        Likelihood is proportional to a product of self._num_observations_per_parameter 2D
        Gaussians and so log likelihood can be computed analytically.

        :param observations: torch.Tensor [batch_size, observation_dim]
            Batch of observations.
        :param parameters: torch.Tensor [batch_size, parameter_dim]
            Batch of parameters.
        :return: torch.Tensor [batch_size]
            Log likelihood log p(x | theta) for each item in the batch.
        """

        if isinstance(parameters, torch.Tensor):
            parameters = utils.tensor2numpy(parameters)

        if isinstance(observations, torch.Tensor):
            observations = utils.tensor2numpy(observations)

        if observations.ndim == 1 and parameters.ndim == 1:
            observations, parameters = (
                observations.reshape(1, -1),
                parameters.reshape(1, -1),
            )

        m0, m1, s0, s1, r = self._unpack_params(parameters)
        logdet = np.log(s0) + np.log(s1) + 0.5 * np.log(1.0 - r ** 2)

        observations = observations.reshape(
            [observations.shape[0], self._num_observations_per_parameter, 2]
        )
        us = np.empty_like(observations)

        us[:, :, 0] = (observations[:, :, 0] - m0) / s0
        us[:, :, 1] = (observations[:, :, 1] - m1 - s1 * r * us[:, :, 0]) / (
            s1 * np.sqrt(1.0 - r ** 2)
        )
        us = us.reshape([us.shape[0], 2 * self._num_observations_per_parameter])

        L = (
            np.sum(scipy.stats.norm.logpdf(us), axis=1)
            - self._num_observations_per_parameter * logdet[:, 0]
        )

        return L

    @staticmethod
    def _unpack_params(parameters):
        """
        Utility function t unpack parameters parameters to m0, m1, s0, s1, r.

        :param parameters: np.array [batch_size, parameter_dim]
            Batch of parameters.
        :return: tuple(np.array)
            Tuple of parameters where each np.array holds a single parameter for the batch.
        """

        assert parameters.shape[1] == 5, "wrong size"

        m0 = parameters[:, [0]]
        m1 = parameters[:, [1]]
        s0 = parameters[:, [2]] ** 2
        s1 = parameters[:, [3]] ** 2
        r = np.tanh(parameters[:, [4]])

        return m0, m1, s0, s1, r

    @staticmethod
    def get_ground_truth_parameters():
        """
        Returns ground truth parameters as specified in 'Sequential Neural Likelihood'.

        :return: torch.Tensor [parameter_dim]
            Ground truth parameters used to generate an observation.
        """
        return torch.Tensor([-0.7, -2.9, -1.0, -0.9, 0.6])

    def get_ground_truth_observation(self):
        """
        Returns ground truth observation using same seed as 'Sequential Neural Likelihood'.

        :return: torch.Tensor [observation_dim]
            Ground truth observation generated by ground truth parameters.
        """
        mean, std = self._get_observation_normalization_parameters()
        ground_truth = torch.Tensor(
            [
                -0.97071232,
                -2.94612244,
                -0.44947218,
                -3.42318484,
                -0.13285634,
                -3.36401699,
                -0.85367595,
                -2.42716377,
            ]
        )
        return (ground_truth - mean) / std

    def get_ground_truth(self):
        """
        :return: (torch.Tensor, torch.Tensor)
            Ground truth (parameter, observation) pair.
        """
        return self.get_ground_truth_parameters(), self.get_ground_truth_observation()

    def get_ground_truth_posterior_samples(self, num_samples=None):
        """
        We have pre-generated posterior samples using MCMC on the product of the analytic
        likelihood and a uniform prior on [-3, 3]^5.
        Thus they are ground truth as long as MCMC has behaved well.
        We load these once if samples have not been loaded before, store them for future use,
        and return as many as are requested.

        :param num_samples: int
            Number of sample to return.
        :return: torch.Tensor [num_samples, parameter_dim]
            Batch of posterior samples.
        """
        if self._posterior_samples is None:
            self._posterior_samples = torch.Tensor(
                np.load(
                    os.path.join(
                        utils.get_data_root(),
                        "nonlinear-gaussian",
                        "true-posterior-samples.npy",
                    )
                )
            )
        if num_samples is not None:
            return self._posterior_samples[:num_samples]
        else:
            return self._posterior_samples

    @property
    def parameter_dim(self):
        return 5

    @property
    def observation_dim(self):
        return 8

    @property
    def name(self):
        return "nonlinear-gaussian"

    @property
    def parameter_plotting_limits(self):
        return [-4, 4]

    @property
    def normalization_parameters(self):
        # mean = torch.zeros(5)
        # std = ((36 / 12) ** 0.5) * torch.ones(5)
        mean = torch.zeros(5)
        std = torch.ones(5)
        return mean, std

    def _get_observation_normalization_parameters(self):
        # mean = torch.Tensor(
        #     [-0.0055, 0.0013, 0.0030, 0.0054, -0.0074, -0.0003, -0.0007, -0.0042]
        # )
        # std = torch.Tensor(
        #     [4.3816, 4.3758, 4.3808, 4.3718, 4.3789, 4.3727, 4.3801, 4.3822]
        # )
        mean = torch.zeros(8)
        std = torch.ones(8)
        return mean, std


def sample_true_posterior():
    prior = distributions.Uniform(low=-3 * torch.ones(5), high=3 * torch.ones(5))
    # print(log_prob)
    potential_function = (
        lambda parameters: simulator.log_prob(
            observations=true_observation, parameters=parameters
        )
        + prior.log_prob(torch.Tensor(parameters)).sum().item()
    )
    sampler = SliceSampler(x=true_parameters, lp_f=potential_function, thin=10)
    sampler.gen(200)
    samples = sampler.gen(2500)
    # figure = corner.corner(
    #     samples,
    #     truths=true_parameters,
    #     truth_color='C1',
    #     bins=25,
    #     color='black',
    #     labels=[r'$ \theta_{1} $', r'$ \theta_{2} $', r'$ \theta_{3} $',
    #             r'$ \theta_{4} $', r'$ \theta_{5} $'],
    #     show_titles=True,
    #     hist_kwargs={'color': 'grey', 'fill': True},
    #     title_fmt='.2f',
    #     plot_contours=True,
    #     quantiles=[0.5]
    # )
    # plt.tight_layout()
    figure = utils.plot_hist_marginals(
        samples, ground_truth=true_parameters, lims=[-4, 4]
    )
    np.save(
        os.path.join(utils.get_output_root(), "./true-posterior-samples-gaussian.npy"),
        samples,
    )
    plt.show()


def main():
    pass


# if __name__ == "__main__":
#     simulator = NonlinearGaussian()
#     samples = simulator.get_ground_truth_posterior_samples()
#     samples = utils.tensor2numpy(samples)
#     figure = utils.plot_hist_marginals(
#         data=samples,
#         ground_truth=utils.tensor2numpy(
#             simulator.get_ground_truth_parameters()
#         ).reshape(-1),
#         lims=[-4, 4],
#     )
#     plt.show()
# true_parameters, true_observation = simulator.get_ground_truth()
# log_prob = simulator.log_prob(true_observation, true_parameters)
# print(log_prob)
# import torch
# from torch import distributions
# import utils

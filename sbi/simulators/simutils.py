import torch

import sbi.simulators as simulators

from torch import distributions

from pyknos import distributions as distributions_


def simulation_wrapper(simulator, parameter_sample_fn, num_samples):

    # if (
    #     isinstance(simulator, simulators.LotkaVolterraSimulator)
    #     and not simulator._has_been_used
    # ):
    #     # if False:
    #     parameters, observations = simulator._get_prior_parameters_observations()
    #     return (
    #         torch.Tensor(parameters)[:num_samples],
    #         torch.Tensor(observations)[:num_samples],
    #     )
    #
    # else:
    #     num_remaining_samples = num_samples
    #     parameters, observations = [], []
    #
    #     while num_remaining_samples > 0:
    #
    #         proposed_parameters = parameter_sample_fn(num_remaining_samples)
    #         proposed_observations = simulator.simulate(proposed_parameters)
    #
    #         for parameter, observation in zip(
    #             proposed_parameters, proposed_observations
    #         ):
    #             if observation is not None:
    #                 parameters.append(parameter.reshape(1, -1))
    #                 observations.append(observation.reshape(1, -1))
    #
    #         num_remaining_samples = num_samples - len(parameters)
    #
    #     return torch.cat(parameters), torch.cat(observations)

    if isinstance(simulator, simulators.LotkaVolterraSimulator):

        if not simulator._has_been_used:
            parameters, observations = simulator._get_prior_parameters_observations()
            return (
                torch.Tensor(parameters)[:num_samples],
                torch.Tensor(observations)[:num_samples],
            )

        else:
            num_remaining_samples = num_samples
            parameters, observations = [], []

            while num_remaining_samples > 0:

                proposed_parameters = parameter_sample_fn(num_remaining_samples)
                proposed_observations = simulator.simulate(proposed_parameters)

                for parameter, observation in zip(
                    proposed_parameters, proposed_observations
                ):
                    if observation is not None:
                        parameters.append(parameter.reshape(1, -1))
                        observations.append(torch.Tensor(observation.reshape(1, -1)))

                num_remaining_samples = num_samples - len(parameters)

            return torch.cat(parameters), torch.cat(observations)

    else:
        parameters = parameter_sample_fn(num_samples)
        observations = simulator.simulate(parameters)
        return torch.Tensor(parameters), torch.Tensor(observations)


def get_simulator_and_prior(task):

    if task == "nonlinear-gaussian":
        simulator = simulators.NonlinearGaussianSimulator()
        prior = distributions.Uniform(
            low=-3 * torch.ones(simulator.parameter_dim),
            high=3 * torch.ones(simulator.parameter_dim),
        )

    elif task == "nonlinear-gaussian-gaussian":
        simulator = simulators.NonlinearGaussianSimulator()
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(5), covariance_matrix=torch.eye(5)
        )

    elif task == "two-moons":
        simulator = simulators.TwoMoonsSimulator()
        a = 2
        prior = distributions.Uniform(
            low=-a * torch.ones(simulator.parameter_dim),
            high=a * torch.ones(simulator.parameter_dim),
        )

    elif task == "linear-gaussian":
        dim, std = 20, 0.5
        simulator = simulators.LinearGaussianSimulator(dim=dim, std=std)
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(dim), covariance_matrix=torch.eye(dim)
        )

    elif task == "lotka-volterra":
        simulator = simulators.LotkaVolterraSimulator(
            summarize_observations=True, gaussian_prior=False
        )
        prior = distributions.Uniform(
            low=-5 * torch.ones(simulator.parameter_dim),
            high=2 * torch.ones(simulator.parameter_dim),
        )

    elif task == "lotka-volterra-gaussian":
        simulator = simulators.LotkaVolterraSimulator(
            summarize_observations=True, gaussian_prior=True
        )
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(4), covariance_matrix=2 * torch.eye(4)
        )

    elif task == "mg1":
        simulator = simulators.MG1Simulator()
        prior = distributions_.MG1Uniform(
            low=torch.zeros(3), high=torch.Tensor([10.0, 10.0, 1.0 / 3.0])
        )

    else:
        raise ValueError(f"'{task}' simulator choice not understood.")

    return simulator, prior

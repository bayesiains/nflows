import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import distributions

import pyknos.utils.plot as plot


class TweakedUniform(distributions.Uniform):
    def log_prob(self, value, context):
        return torchutils.sum_except_batch(super().log_prob(value))
        # result = super().log_prob(value)
        # if len(result.shape) == 2 and result.shape[1] == 1:
        #     return result.reshape(-1)
        # else:
        #     return result

    def sample(self, num_samples, context):
        return super().sample((num_samples,))


class MG1Uniform(distributions.Uniform):
    def log_prob(self, value):
        return super().log_prob(self._to_noise(value))

    def sample(self, sample_shape=torch.Size()):
        return self._to_parameters(super().sample(sample_shape))

    def _to_parameters(self, noise):
        A_inv = torch.Tensor([[1.0, 1, 0], [0, 1, 0], [0, 0, 1]])
        return noise @ A_inv

    def _to_noise(self, parameters):
        A = torch.Tensor([[1.0, -1, 0], [0, 1, 0], [0, 0, 1]])
        return parameters @ A


class LotkaVolterraOscillating:
    def __init__(self):
        mean = torch.log(torch.Tensor([0.01, 0.5, 1, 0.01]))
        sigma = 0.5
        covariance = sigma ** 2 * torch.eye(4)
        self._gaussian = distributions.MultivariateNormal(
            loc=mean, covariance_matrix=covariance
        )
        self._uniform = distributions.Uniform(
            low=-5 * torch.ones(4), high=2 * torch.ones(4)
        )
        self._log_normalizer = -torch.log(
            torch.erf((2 - mean) / sigma) - torch.erf((-5 - mean) / sigma)
        ).sum()

    def log_prob(self, value):
        unnormalized_log_prob = self._gaussian.log_prob(value) + self._uniform.log_prob(
            value
        ).sum(-1)
        return self._log_normalizer + unnormalized_log_prob

    def sample(self, sample_shape=torch.Size()):
        num_remaining_samples = sample_shape[0]
        samples = []
        while num_remaining_samples > 0:
            candidate_samples = self._gaussian.sample((num_remaining_samples,))

            uniform_log_prob = self._uniform.log_prob(candidate_samples).sum(-1)

            accepted_samples = candidate_samples[~torch.isinf(uniform_log_prob)]
            samples.append(accepted_samples.detach())

            num_accepted = (~torch.isinf(uniform_log_prob)).sum().item()
            num_remaining_samples -= num_accepted

        # Aggregate collected samples.
        samples = torch.cat(samples)

        # Make sure we have the right amount.
        samples = samples[: sample_shape[0], ...]
        assert samples.shape[0] == sample_shape[0]

        return samples


def _test():
    # prior = MG1Uniform(low=torch.zeros(3), high=torch.Tensor([10, 10, 1 / 3]))
    # uniform = distributions.Uniform(
    #     low=torch.zeros(3), high=torch.Tensor([10, 10, 1 / 3])
    # )
    # x = torch.Tensor([10, 20, 1 / 3]).reshape(1, -1)
    # print(uniform.log_prob(x))
    # print(prior.log_prob(x))
    d = LotkaVolterraOscillating()
    samples = d.sample((1000,))
    plot.plot_hist_marginals(utils.tensor2numpy(samples), lims=[-6, 3])
    plt.show()
    # w, x, y, z = np.meshgrid(
    #     np.linspace(-5, 2, 100),
    #     np.linspace(-5, 2, 100),
    #     np.linspace(-5, 2, 100),
    #     np.linspace(-5, 2, 100),
    # )
    # print(w.shape)
    # data = np.concatenate(
    #     (w.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)),
    #     axis=-1,
    # )
    # print(data.shape)
    # print(samples.shape)


def main():
    _test()


if __name__ == "__main__":
    main()

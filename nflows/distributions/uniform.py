from typing import Union

import torch
from torch import distributions

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class BoxUniform(Distribution):
    def __init__(
        self,
        low: Union[torch.Tensor, float],
        high: Union[torch.Tensor, float]    
    ):
        """Multidimensionqal uniform distribution defined on a box.
            
        Args:
            low (Tensor or float): lower range (inclusive).
            high (Tensor or float): upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
        """
        super().__init__()
        if low.shape != high.shape:
            raise ValueError(
                "low and high are not of the same size"
            )

        if not (low < high).byte().all():
            raise ValueError(
                "low has elements that are higher than high"
            )

        self._shape = low.shape
        self._low = low
        self._high = high
        self._log_prob_value = -torch.sum(torch.log(high - low))

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return self._log_prob_value.expand(inputs.shape[0])

    def _sample(self, num_samples, context):   
        context_size = 1 if context is None else context.shape[0]
        low_expanded =  self._low.expand(context_size  * num_samples, *self._shape)
        high_expanded = self._high.expand(context_size * num_samples, *self._shape)
        samples = low_expanded + torch.rand(context_size * num_samples, *self._shape)*(high_expanded - low_expanded)

        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(samples, [context_size, num_samples])
        
class MG1Uniform(distributions.Uniform):
    def log_prob(self, value):
        return super().log_prob(self._to_noise(value))

    def sample(self, sample_shape=torch.Size()):
        return self._to_parameters(super().sample(sample_shape))

    def _to_parameters(self, noise):
        A_inv = torch.tensor([[1.0, 1, 0], [0, 1, 0], [0, 0, 1]])
        return noise @ A_inv

    def _to_noise(self, parameters):
        A = torch.tensor([[1.0, -1, 0], [0, 1, 0], [0, 0, 1]])
        return parameters @ A


class LotkaVolterraOscillating:
    def __init__(self):
        mean = torch.log(torch.tensor([0.01, 0.5, 1, 0.01]))
        sigma = 0.5
        covariance = sigma ** 2 * torch.eye(4)
        self._gaussian = distributions.MultivariateNormal(
            loc=mean, covariance_matrix=covariance
        )
        self._uniform = BoxUniform(low=-5 * torch.ones(4), high=2 * torch.ones(4))
        self._log_normalizer = -torch.log(
            torch.erf((2 - mean) / sigma) - torch.erf((-5 - mean) / sigma)
        ).sum()

    def log_prob(self, value):
        unnormalized_log_prob = self._gaussian.log_prob(value) + self._uniform.log_prob(
            value
        )

        return self._log_normalizer + unnormalized_log_prob

    def sample(self, sample_shape=torch.Size()):
        num_remaining_samples = sample_shape[0]
        samples = []
        while num_remaining_samples > 0:
            candidate_samples = self._gaussian.sample((num_remaining_samples,))

            uniform_log_prob = self._uniform.log_prob(candidate_samples)

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

from typing import Union

import torch
from torch import distributions


class BoxUniform(distributions.Independent):
    def __init__(
        self,
        low: Union[torch.Tensor, float],
        high: Union[torch.Tensor, float],
        reinterpreted_batch_ndims: int = 1,
    ):
        """Multidimensionqal uniform distribution defined on a box.
        
        A `Uniform` distribution initialized with e.g. a parameter vector low or high of length 3 will result in a /batch/ dimension of length 3. A log_prob evaluation will then output three numbers, one for each of the independent Uniforms in the batch. Instead, a `BoxUniform` initialized in the same way has three /event/ dimensions, and returns a scalar log_prob corresponding to whether the evaluated point is in the box defined by low and high or outside. 
    
        Refer to torch.distributions.Uniform and torch.distributions.Independent for further documentation.
    
        Args:
            low (Tensor or float): lower range (inclusive).
            high (Tensor or float): upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
        """

        super().__init__(
            distributions.Uniform(low=low, high=high), reinterpreted_batch_ndims
        )


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

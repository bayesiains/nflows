"""
Implementation of models based on
C. M. Bishop, "Mixture Density Networks", NCRG Report (1994)
"""

import numpy as np
import torch
import sbi.utils as utils

from torch import nn
from torch.nn import functional as F


class MultivariateGaussianMDN(nn.Module):
    """
    Implementation of
    'Mixture Density Networks'
    Bishop
    Neural Computing Research Group Report 1994
    https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf

    Mixture family is multivariate Gaussian with full (rather than diagonal)
    covariance matrices.
    """

    def __init__(
        self,
        features,
        context_features,
        hidden_features,
        hidden_net,
        num_components,
        custom_initialization=False,
    ):
        """
        Parameters
        ----------
        :param input_dim: int
            Dimension of inputs.
        :param hidden_dim: int
            Dimension of final layer of hidden net.
        :param hidden_net:
            nets.ModuleNetwork which outputs final hidden representation before
            paramterization layers (i.e logits, means, and log precisions).
        :param num_components: int
            Number of mixture components.
        :param output_dim: int
            Dimension of output density.
        """

        super().__init__()

        self._features = features
        self._context_features = context_features
        self._hidden_features = hidden_features
        self._num_components = num_components
        self._num_upper_params = (features * (features - 1)) // 2

        self._row_ix, self._column_ix = np.triu_indices(features, k=1)
        self._diag_ix = range(features)

        # Modules
        self._hidden_net = hidden_net

        self._logits_layer = nn.Linear(hidden_features, num_components)

        self._means_layer = nn.Linear(hidden_features, num_components * features)

        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )
        self._upper_layer = nn.Linear(
            hidden_features, num_components * self._num_upper_params
        )

        # Constant for numerical stability.
        self._epsilon = 1e-2

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

    def get_mixture_components(self, context):
        """
        :param context: torch.Tensor [batch_size, input_dim]
            The input to the MDN.
        :return: tuple(
            torch Tensor [batch_size, n_mixtures],
            torch.Tensor [batch_size, n_mixtures, output_dim],
            torch.Tensor [batch_size, n_mixtures, output_dim, output_dim],
            torch.Tensor [1],
            torch.Tensor [batch_size, n_mixtures, output_dim, output_dim]
            )
            Tuple containing logits, means, precisions,
            sum of log diagonal of precision factors, and precision factors themselves.
            Recall upper triangular precision factor A such that SIGMA^-1 = A^T A.
        """

        h = self._hidden_net(context)

        # Logits and Means are unconstrained and are obtained directly from the
        # output of a linear layer.
        logits = self._logits_layer(h)
        means = self._means_layer(h).view(-1, self._num_components, self._features)

        # Unconstrained diagonal and upper triangular quantities are unconstrained.
        unconstrained_diagonal = self._unconstrained_diagonal_layer(h).view(
            -1, self._num_components, self._features
        )
        upper = self._upper_layer(h).view(
            -1, self._num_components, self._num_upper_params
        )

        # Elements of diagonal of precision factor must be positive
        # (recall precision factor A such that SIGMA^-1 = A^T A).
        diagonal = F.softplus(unconstrained_diagonal) + self._epsilon

        # Create empty precision factor matrix, and fill with appropriate quantities.
        precision_factors = torch.zeros(
            means.shape[0], self._num_components, self._features, self._features
        )
        precision_factors[..., self._diag_ix, self._diag_ix] = diagonal
        precision_factors[..., self._row_ix, self._column_ix] = upper

        # Precisions are given by SIGMA^-1 = A^T A.
        precisions = torch.matmul(
            torch.transpose(precision_factors, 2, 3), precision_factors
        )

        # The sum of the log diagonal of A is used in the likelihood calculation.
        sumlogdiag = torch.sum(torch.log(diagonal), dim=-1)

        return logits, means, precisions, sumlogdiag, precision_factors

    def log_prob(self, inputs, context=None):
        """
        Evaluates log p(inputs | context), where p is a multivariate mixture of Gaussians
        with mixture coefficients, means, and precisions given as a neural network function.

        :param inputs: torch.Tensor [batch_size, input_dim]
            Input variable.
        :param context: torch.Tensor [batch_size, context_dim]
            Conditioning variable.
        :return: torch.Tensor [1]
            Log probability of inputs given context under model.
        """

        # Get necessary quantities.
        logits, means, precisions, sumlogdiag, _ = self.get_mixture_components(context)

        batch_size, n_mixtures, output_dim = means.size()
        inputs = inputs.view(-1, 1, output_dim)

        # Split up evaluation into parts.
        a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        b = -(output_dim / 2.0) * np.log(2 * np.pi)
        c = sumlogdiag
        d1 = (inputs.expand_as(means) - means).view(
            batch_size, n_mixtures, output_dim, 1
        )
        d2 = torch.matmul(precisions, d1)
        d = -0.5 * torch.matmul(torch.transpose(d1, 2, 3), d2).view(
            batch_size, n_mixtures
        )

        return torch.logsumexp(a + b + c + d, dim=-1)

    def sample(self, num_samples, context):
        """
        Generated num_samples independent samples from p(inputs | context).
        NB: Generates num_samples samples for EACH item in context batch i.e. returns
        (num_samples * batch_size) samples in total.

        :param num_samples: int
            Number of samples to generate.
        :param context: torch.Tensor [batch_size, context_dim]
            Conditioning variable.
        :return: torch.Tensor [batch_size, num_samples, output_dim]
            Batch of generated samples.
        """

        # Get necessary quantities.
        logits, means, _, _, precision_factors = self.get_mixture_components(context)
        batch_size, n_mixtures, output_dim = means.shape

        # We need (batch_size * num_samples) samples in total.
        means, precision_factors = (
            utils.repeat_rows(means, num_samples),
            utils.repeat_rows(precision_factors, num_samples),
        )

        # Normalize the logits for the coefficients.
        coefficients = F.softmax(logits, dim=-1)  # [batch_size, num_components]

        # Choose num_samples mixture components per example in the batch.
        choices = torch.multinomial(
            coefficients, num_samples=num_samples, replacement=True
        ).view(
            -1
        )  # [batch_size, num_samples]

        # Create dummy index for indexing means and precision factors.
        ix = utils.repeat_rows(torch.arange(batch_size), num_samples)

        # Select means and precision factors.
        chosen_means = means[ix, choices, :]
        chosen_precision_factors = precision_factors[ix, choices, :, :]

        # Batch triangular solve to multiply standard normal samples by inverse
        # of upper triangular precision factor.
        zero_mean_samples, _ = torch.triangular_solve(
            torch.randn(
                batch_size * num_samples, output_dim, 1
            ),  # Need dummy final dimension.
            chosen_precision_factors,
        )

        # Mow center samples at chosen means, removing dummy final dimension
        # from triangular solve.
        samples = chosen_means + zero_mean_samples.squeeze(-1)

        return samples.reshape(batch_size, num_samples, output_dim)

    def _initialize(self):
        """
        Initializes MDN so that mixture coefficients are approximately uniform,
        and covariances are approximately the identity.

        :return: None
        """
        # Initialize mixture coefficients to near uniform.
        self._logits_layer.weight.data = self._epsilon * torch.randn(
            self._num_components, self._hidden_features
        )
        self._logits_layer.bias.data = self._epsilon * torch.randn(self._num_components)

        # Initialize diagonal of precision factors to inverse of softplus at 1.
        self._unconstrained_diagonal_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._features, self._hidden_features
        )
        self._unconstrained_diagonal_layer.bias.data = torch.log(
            torch.exp(torch.Tensor([1 - self._epsilon])) - 1
        ) * torch.ones(
            self._num_components * self._features
        ) + self._epsilon * torch.randn(
            self._num_components * self._features
        )

        # Initialize off-diagonal of precision factors to zero.
        self._upper_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params, self._hidden_features
        )
        self._upper_layer.bias.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params
        )


def main():
    # probs = torch.Tensor([[1, 0], [0, 1]])
    # samples = torch.multinomial(probs, num_samples=5, replacement=True)
    # print(samples)
    # quit()
    mdn = MultivariateGaussianMDN(
        features=2,
        context_features=3,
        hidden_features=16,
        hidden_net=nn.Linear(3, 16),
        num_components=4,
    )
    inputs = torch.randn(1, 3)
    samples = mdn.sample(9, inputs)
    print(samples.shape)


if __name__ == "__main__":
    main()

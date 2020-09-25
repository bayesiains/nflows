import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.linear import Linear
from nflows.transforms.orthogonal import HouseholderSequence


class SVDLinear(Linear):
    """A linear module using the SVD decomposition for the weight matrix."""

    def __init__(
        self, features, num_householder, using_cache=False, identity_init=True, eps=1e-3
    ):
        super().__init__(features, using_cache)

        assert num_householder % 2 == 0

        # minimum value for diagonal
        self.eps = eps
        # First orthogonal matrix (U).
        self.orthogonal_1 = HouseholderSequence(
            features=features, num_transforms=num_householder
        )

        # Logs of diagonal entries of the diagonal matrix (S).
        self.unconstrained_diagonal = nn.Parameter(torch.zeros(features))

        # Second orthogonal matrix (V^T).
        self.orthogonal_2 = HouseholderSequence(
            features=features, num_transforms=num_householder
        )

        self.identity_init = identity_init
        self._initialize()

    @property
    def diagonal(self):
        return self.eps + F.softplus(self.unconstrained_diagonal)

    @property
    def log_diagonal(self):
        return torch.log(self.diagonal)

    def _initialize(self):
        init.zeros_(self.bias)
        if self.identity_init:
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_diagonal, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.unconstrained_diagonal, -stdv, stdv)

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs, _ = self.orthogonal_2(inputs)  # Ignore logabsdet as we know it's zero.
        outputs *= self.diagonal
        outputs, _ = self.orthogonal_1(
            outputs
        )  # Ignore logabsdet as we know it's zero.
        outputs += self.bias

        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal_1.inverse(
            outputs
        )  # Ignore logabsdet since we know it's zero.
        outputs /= self.diagonal
        outputs, _ = self.orthogonal_2.inverse(
            outputs
        )  # Ignore logabsdet since we know it's zero.
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal = torch.diag(self.diagonal)
        weight, _ = self.orthogonal_2.inverse(diagonal)
        weight, _ = self.orthogonal_1(weight.t())
        return weight.t()

    def weight_inverse(self):
        """Cost:
            inverse = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal_inv = torch.diag(torch.reciprocal(self.diagonal))
        weight_inv, _ = self.orthogonal_1(diagonal_inv)
        weight_inv, _ = self.orthogonal_2.inverse(weight_inv.t())
        return weight_inv.t()

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_diagonal)

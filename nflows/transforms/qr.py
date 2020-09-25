import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.linear import Linear
from nflows.transforms.orthogonal import HouseholderSequence


class QRLinear(Linear):
    """A linear module using the QR decomposition for the weight matrix."""

    def __init__(self, features, num_householder, using_cache=False):
        super().__init__(features, using_cache)

        # Parameterization for R
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)
        n_triangular_entries = ((features - 1) * features) // 2
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.log_upper_diag = nn.Parameter(torch.zeros(features))

        # Parameterization for Q
        self.orthogonal = HouseholderSequence(
            features=features, num_transforms=num_householder
        )

        self._initialize()

    def _initialize(self):
        stdv = 1.0 / np.sqrt(self.features)
        init.uniform_(self.upper_entries, -stdv, stdv)
        init.uniform_(self.log_upper_diag, -stdv, stdv)
        init.constant_(self.bias, 0.0)

    def _create_upper(self):
        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = torch.exp(
            self.log_upper_diag
        )
        return upper

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N + KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        upper = self._create_upper()

        outputs = F.linear(inputs, upper)
        outputs, _ = self.orthogonal(outputs)  # Ignore logabsdet as we know it's zero.
        outputs += self.bias

        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N + KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        upper = self._create_upper()
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal.inverse(
            outputs
        )  # Ignore logabsdet since we know it's zero.
        outputs, _ = torch.triangular_solve(outputs.t(), upper, upper=True)
        outputs = outputs.t()
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
        upper = self._create_upper()
        weight, _ = self.orthogonal(upper.t())
        return weight.t()

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3 + KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        upper = self._create_upper()
        identity = torch.eye(self.features, self.features)
        upper_inv, _ = torch.triangular_solve(identity, upper, upper=True)
        weight_inv, _ = self.orthogonal(upper_inv)
        return weight_inv

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_upper_diag)

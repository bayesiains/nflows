"""Implementations of linear transforms."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nflows.transforms.base import Transform
from nflows.utils import torchutils
import nflows.utils.typechecks as check


class LinearCache:
    """Helper class to store the cache of a linear transform.

    The cache consists of: the weight matrix, its inverse and its log absolute determinant.
    """

    def __init__(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None


class Linear(Transform):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        # Caching flag and values.
        self.using_cache = using_cache
        self.cache = LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()

        elif self.cache.weight is None:
            self.cache.weight = self.weight()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = (-self.cache.logabsdet) * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            (
                self.cache.inverse,
                self.cache.logabsdet,
            ) = self.weight_inverse_and_logabsdet()

        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode=True):
        if mode:
            # If training again, invalidate cache.
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        if not check.is_bool(mode):
            raise TypeError("Mode must be boolean.")
        self.using_cache = mode

    def weight_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # and its logabsdet together.
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # inverse and weight matrix logabsdet together.
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(self, inputs):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self):
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self):
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self):
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


class NaiveLinear(Linear):
    """A general linear transform that uses an unconstrained weight matrix.

    This transform explicitly computes the log absolute determinant in the forward direction
    and uses a linear solver in the inverse direction.

    Both forward and inverse directions have a cost of O(D^3), where D is the dimension
    of the input.
    """

    def __init__(self, features, orthogonal_initialization=True, using_cache=False):
        """Constructor.

        Args:
            features: int, number of input features.
            orthogonal_initialization: bool, if True initialize weights to be a random
                orthogonal matrix.

        Raises:
            TypeError: if `features` is not a positive integer.
        """
        super().__init__(features, using_cache)

        if orthogonal_initialization:
            self._weight = nn.Parameter(torchutils.random_orthogonal(features))
        else:
            self._weight = nn.Parameter(torch.empty(features, features))
            stdv = 1.0 / np.sqrt(features)
            init.uniform_(self._weight, -stdv, stdv)

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = F.linear(inputs, self._weight, self.bias)
        logabsdet = torchutils.logabsdet(self._weight)
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^3 + D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = inputs - self.bias
        outputs, lu = torch.solve(outputs.t(), self._weight)  # Linear-system solver.
        outputs = outputs.t()
        # The linear-system solver returns the LU decomposition of the weights, which we
        # can use to obtain the log absolute determinant directly.
        logabsdet = -torch.sum(torch.log(torch.abs(torch.diag(lu))))
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(1)
        """
        return self._weight

    def weight_inverse(self):
        """
        Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        return torch.inverse(self._weight)

    def weight_inverse_and_logabsdet(self):
        """
        Cost:
            inverse = O(D^3)
            logabsdet = O(D)
        where:
            D = num of features
        """
        # If both weight inverse and logabsdet are needed, it's cheaper to compute both together.
        identity = torch.eye(self.features, self.features)
        weight_inv, lu = torch.solve(identity, self._weight)  # Linear-system solver.
        logabsdet = torch.sum(torch.log(torch.abs(torch.diag(lu))))
        return weight_inv, logabsdet

    def logabsdet(self):
        """Cost:
            logabsdet = O(D^3)
        where:
            D = num of features
        """
        return torchutils.logabsdet(self._weight)

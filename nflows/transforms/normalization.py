"""Implementation of normalization-based transforms."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nflows.transforms.base import InverseNotAvailable, Transform
import nflows.utils.typechecks as check

# class BatchNorm(Transform):
#     """Transform that performs batch normalization.
#
#     Limitations:
#         * It works only for 1-dim inputs.
#         * Inverse is not available in training mode, only in eval mode.
#     """
#
#     def __init__(self, features, eps=1e-5, momentum=0.1, affine=True):
#         if not check.is_positive_int(features):
#             raise TypeError('Number of features must be a positive integer.')
#         super().__init__()
#
#         self.batch_norm = nets.BatchNorm1d(
#             num_features=features,
#             eps=eps,
#             momentum=momentum,
#             affine=affine,
#             track_running_stats=True,  # We mustn't use batch statistics in eval mode.
#         )
#
#     def forward(self, inputs):
#         if inputs.dim() != 2:
#             raise ValueError('Expected 2-dim inputs, got inputs of shape: {}'.format(inputs.shape))
#
#         outputs = self.batch_norm(inputs)
#
#         if self.training:
#             var = torch.var(inputs, dim=0, unbiased=False)
#         else:
#             var = self.batch_norm.running_var
#         logabsdet = -0.5 * torch.log(var + self.batch_norm.eps)
#         if self.batch_norm.affine:
#             logabsdet += torch.log(self.batch_norm.weight)
#         logabsdet = torch.sum(logabsdet)
#         logabsdet = logabsdet * torch.ones(inputs.shape[0])
#
#         return outputs, logabsdet
#
#     def inverse(self, inputs):
#         if self.training:
#             raise InverseNotAvailable(
#                 'Batch norm inverse is only available in eval mode, not in training mode.')
#         if inputs.dim() != 2:
#             raise ValueError('Expected 2-dim inputs, got inputs of shape: {}'.format(inputs.shape))
#
#         outputs = inputs.clone()
#         if self.batch_norm.affine:
#             outputs -= self.batch_norm.bias
#             outputs /= self.batch_norm.weight
#         outputs *= torch.sqrt(self.batch_norm.running_var + self.batch_norm.eps)
#         outputs += self.batch_norm.running_mean
#
#         logabsdet = 0.5 * torch.log(self.batch_norm.running_var + self.batch_norm.eps)
#         if self.batch_norm.affine:
#             logabsdet -= torch.log(self.batch_norm.weight)
#         logabsdet = torch.sum(logabsdet)
#         logabsdet = logabsdet * torch.ones(inputs.shape[0])
#
#         return outputs, logabsdet


class BatchNorm(Transform):
    """Transform that performs batch normalization.

    Limitations:
        * It works only for 1-dim inputs.
        * Inverse is not available in training mode, only in eval mode.
    """

    def __init__(self, features, eps=1e-5, momentum=0.1, affine=True):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.momentum = momentum
        self.eps = eps
        constant = np.log(np.exp(1 - eps) - 1)
        self.unconstrained_weight = nn.Parameter(constant * torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.zeros(features))

    @property
    def weight(self):
        return F.softplus(self.unconstrained_weight) + self.eps

    def forward(self, inputs, context=None):
        if inputs.dim() != 2:
            raise ValueError(
                "Expected 2-dim inputs, got inputs of shape: {}".format(inputs.shape)
            )

        if self.training:
            mean, var = inputs.mean(0), inputs.var(0)
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean, var = self.running_mean, self.running_var

        outputs = (
            self.weight * ((inputs - mean) / torch.sqrt((var + self.eps))) + self.bias
        )

        logabsdet_ = torch.log(self.weight) - 0.5 * torch.log(var + self.eps)
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0])

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if self.training:
            raise InverseNotAvailable(
                "Batch norm inverse is only available in eval mode, not in training mode."
            )
        if inputs.dim() != 2:
            raise ValueError(
                "Expected 2-dim inputs, got inputs of shape: {}".format(inputs.shape)
            )

        outputs = (
            torch.sqrt(self.running_var + self.eps)
            * ((inputs - self.bias) / self.weight)
            + self.running_mean
        )

        logabsdet_ = -torch.log(self.weight) + 0.5 * torch.log(
            self.running_var + self.eps
        )
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0])

        return outputs, logabsdet


class ActNorm(Transform):
    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        if self.training and not self.initialized:
            self._initialize(inputs)

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = scale * inputs + shift

        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        else:
            batch_size, _ = inputs.shape
            logabsdet = torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (inputs - shift) / scale

        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = -h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        else:
            batch_size, _ = inputs.shape
            logabsdet = -torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)

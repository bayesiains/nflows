"""Implementations of invertible non-linearities."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nflows.transforms import splines
from nflows.transforms.base import (
    CompositeTransform,
    InputOutsideDomain,
    InverseTransform,
    Transform,
)
from nflows.utils import torchutils


class Tanh(Transform):
    def forward(self, inputs, context=None):
        outputs = torch.tanh(inputs)
        logabsdet = torch.log(1 - outputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
            raise InputOutsideDomain()
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class LogTanh(Transform):
    """Tanh with unbounded output. 

    Constructed by selecting a cut_point, and replacing values to the right of cut_point
    with alpha * log(beta * x), and to the left of -cut_point with -alpha * log(-beta *
    x). alpha and beta are set to match the value and the first derivative of tanh at
    cut_point."""

    def __init__(self, cut_point=1):
        if cut_point <= 0:
            raise ValueError("Cut point must be positive.")
        super().__init__()

        self.cut_point = cut_point
        self.inv_cut_point = np.tanh(cut_point)

        self.alpha = (1 - np.tanh(np.tanh(cut_point))) / cut_point
        self.beta = np.exp(
            (np.tanh(cut_point) - self.alpha * np.log(cut_point)) / self.alpha
        )

    def forward(self, inputs, context=None):
        mask_right = inputs > self.cut_point
        mask_left = inputs < -self.cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = torch.tanh(inputs[mask_middle])
        outputs[mask_right] = self.alpha * torch.log(self.beta * inputs[mask_right])
        outputs[mask_left] = self.alpha * -torch.log(-self.beta * inputs[mask_left])

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = torch.log(1 - outputs[mask_middle] ** 2)
        logabsdet[mask_right] = torch.log(self.alpha / inputs[mask_right])
        logabsdet[mask_left] = torch.log(-self.alpha / inputs[mask_left])
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):

        mask_right = inputs > self.inv_cut_point
        mask_left = inputs < -self.inv_cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = 0.5 * torch.log(
            (1 + inputs[mask_middle]) / (1 - inputs[mask_middle])
        )
        outputs[mask_right] = torch.exp(inputs[mask_right] / self.alpha) / self.beta
        outputs[mask_left] = -torch.exp(-inputs[mask_left] / self.alpha) / self.beta

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = -torch.log(1 - inputs[mask_middle] ** 2)
        logabsdet[mask_right] = (
            -np.log(self.alpha * self.beta) + inputs[mask_right] / self.alpha
        )
        logabsdet[mask_left] = (
            -np.log(self.alpha * self.beta) - inputs[mask_left] / self.alpha
        )
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet


class LeakyReLU(Transform):
    def __init__(self, negative_slope=1e-2):
        if negative_slope <= 0:
            raise ValueError("Slope must be positive.")
        super().__init__()
        self.negative_slope = negative_slope
        self.log_negative_slope = torch.log(torch.as_tensor(self.negative_slope))

    def forward(self, inputs, context=None):
        outputs = F.leaky_relu(inputs, negative_slope=self.negative_slope)
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = F.leaky_relu(inputs, negative_slope=(1 / self.negative_slope))
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = -self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class Sigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    def forward(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs)
        )
        return outputs, logabsdet


class Logit(InverseTransform):
    def __init__(self, temperature=1, eps=1e-6):
        super().__init__(Sigmoid(temperature=temperature, eps=eps))


class GatedLinearUnit(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, context=None):
        gate = torch.sigmoid(context)
        # return inputs * (1 + gate), torch.log(torch.ones_like(gate) + gate).reshape(-1)
        return inputs * gate, torch.log(gate).reshape(-1)

    def inverse(self, inputs, context=None):
        gate = torch.sigmoid(context)
        # return inputs / (1 + gate), - torch.log(torch.ones_like(gate) + gate).reshape(-1)
        return inputs / gate, -torch.log(gate).reshape(-1)


class CauchyCDF(Transform):
    def __init__(self, location=None, scale=None, features=None):
        super().__init__()

    def forward(self, inputs, context=None):
        outputs = (1 / np.pi) * torch.atan(inputs) + 0.5
        logabsdet = torchutils.sum_except_batch(
            -np.log(np.pi) - torch.log(1 + inputs ** 2)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        outputs = torch.tan(np.pi * (inputs - 0.5))
        logabsdet = -torchutils.sum_except_batch(
            -np.log(np.pi) - torch.log(1 + outputs ** 2)
        )
        return outputs, logabsdet


class CauchyCDFInverse(InverseTransform):
    def __init__(self, location=None, scale=None, features=None):
        super().__init__(CauchyCDF(location=location, scale=scale, features=features))


class CompositeCDFTransform(CompositeTransform):
    def __init__(self, squashing_transform, cdf_transform):
        super().__init__(
            [squashing_transform, cdf_transform, InverseTransform(squashing_transform),]
        )


def _share_across_batch(params, batch_size):
    return params[None, ...].expand(batch_size, *params.shape)


class PiecewiseLinearCDF(Transform):
    def __init__(self, shape, num_bins=10, tails=None, tail_bound=1.0):
        super().__init__()

        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_pdf = nn.Parameter(torch.randn(*shape, num_bins))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_pdf = _share_across_batch(self.unnormalized_pdf, batch_size)

        if self.tails is None:
            outputs, logabsdet = splines.linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse
            )
        else:
            outputs, logabsdet = splines.unconstrained_linear_spline(
                inputs=inputs,
                unnormalized_pdf=unnormalized_pdf,
                inverse=inverse,
                tails=self.tails,
                tail_bound=self.tail_bound,
            )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseQuadraticCDF(Transform):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        min_bin_width=splines.quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.quadratic.DEFAULT_MIN_BIN_HEIGHT,
    ):
        super().__init__()
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_widths = nn.Parameter(torch.randn(*shape, num_bins))
        if tails is None:
            self.unnormalized_heights = nn.Parameter(torch.randn(*shape, num_bins + 1))
        else:
            self.unnormalized_heights = nn.Parameter(torch.randn(*shape, num_bins - 1))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(
            self.unnormalized_heights, batch_size
        )

        if self.tails is None:
            spline_fn = splines.quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseCubicCDF(Transform):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        min_bin_width=splines.cubic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.cubic.DEFAULT_MIN_BIN_HEIGHT,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_widths = nn.Parameter(torch.randn(*shape, num_bins))
        self.unnormalized_heights = nn.Parameter(torch.randn(*shape, num_bins))
        self.unnorm_derivatives_left = nn.Parameter(torch.randn(*shape, 1))
        self.unnorm_derivatives_right = nn.Parameter(torch.randn(*shape, 1))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(
            self.unnormalized_heights, batch_size
        )
        unnorm_derivatives_left = _share_across_batch(
            self.unnorm_derivatives_left, batch_size
        )
        unnorm_derivatives_right = _share_across_batch(
            self.unnorm_derivatives_right, batch_size
        )

        if self.tails is None:
            spline_fn = splines.cubic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_cubic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseRationalQuadraticCDF(Transform):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        identity_init=False,
        min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.tail_bound = tail_bound
        self.tails = tails

        if isinstance(shape, int):
            shape = (shape,)
        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = (
                (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            )
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            num_derivatives = (
                (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            )
            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(
            self.unnormalized_heights, batch_size
        )
        unnormalized_derivatives = _share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)

"""Implementations of some standard transforms."""

from typing import Optional
import warnings
import numpy as np
import torch
from nflows.utils.torchutils import ensure_tensor
from nflows.transforms.base import Transform


class IdentityTransform(Transform):
    """Transform that leaves input unchanged."""

    def forward(self, inputs: torch.Tensor, context=Optional[torch.Tensor]):
        num_batches = inputs.shape[0]
        logabsdet = torch.zeros(num_batches)
        return inputs, logabsdet

    def inverse(self, inputs: torch.Tensor, context=Optional[torch.Tensor]):
        return self(inputs, context)


class PointwiseAffineTransform(Transform):
    def __init__(
        self,
        shift: torch.Tensor = torch.tensor(0.0),
        scale: torch.Tensor = torch.tensor(1.0),
    ):
        super().__init__()
        shift, scale = map(ensure_tensor, (shift, scale))

        # reject scales < ~0 (upto dtype precision)
        is_scale_positive = torch.isfinite(torch.log(scale)).any()
        if not is_scale_positive:
            raise ValueError("Scale must be strictly positive.")

        self.register_buffer("_shift", shift)
        self.register_buffer("_scale", scale)

    @property
    def _log_scale(self):
        return torch.log(self._scale)

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        num_dims = torch.prod(ensure_tensor(inputs.shape[1:]), dtype=torch.float)
        outputs = inputs * self._scale + self._shift
        logabsdet = torch.full([batch_size], self._log_scale * num_dims)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]
        num_dims = torch.prod(ensure_tensor(inputs.shape[1:]), dtype=torch.float)
        outputs = (inputs - self._shift) / self._scale
        logabsdet = torch.full([batch_size], -self._log_scale * num_dims)
        return outputs, logabsdet


class AffineTransform(PointwiseAffineTransform):
    def __init__(
        self,
        shift: torch.Tensor = torch.tensor(0.0),
        scale: torch.Tensor = torch.tensor(1.0),
    ):

        warnings.warn("Use PointwiseAffineTransform", DeprecationWarning)

        if shift is None:
            warnings.warn("`shift=None` deprecated; default is 0.0")
            shift = torch.tensor(0.0)
        if scale is None:
            warnings.warn("`scale=None` deprecated; default is 1.0.")
            scale = torch.tensor(1.0)

        super().__init__(shift, scale)

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        outputs = inputs * self._scale + self._shift
        logabsdet = self._log_scale.reshape(1, -1).repeat(batch_size, 1).sum(dim=-1)
        return outputs, logabsdet

AffineScalarTransform = AffineTransform

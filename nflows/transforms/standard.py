"""Implementations of some standard transforms."""

from typing import Optional
import warnings
import numpy as np
import torch
from nflows.utils.torchutils import ensure_tensor
from nflows.transforms.base import Transform
from nflows.utils.torchutils import ensure_tensor, numel


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

    # XXX memoize?
    def _batch_logabsdet(self, batch_shape: torch.Size):
        """Return log abs det with input batch shape."""

        if numel(self._log_scale) > 1:
            return self._log_scale.expand(batch_shape).sum()
        else:
            # when log_scale is a scalar, we use n*log_scale, which is more
            # numerically accurate than \sum_1^n log_scale.
            return self._log_scale * numel(batch_shape)

    def forward(self, inputs: torch.Tensor, context=Optional[torch.Tensor]):
        num_batches, *batch_shape = inputs.shape

        # RuntimeError here => shift/scale not broadcastable to input
        outputs = inputs * self._scale + self._shift
        logabsdet = self._batch_logabsdet(batch_shape).expand(num_batches)

        return outputs, logabsdet

    def inverse(self, inputs: torch.Tensor, context=Optional[torch.Tensor]):
        num_batches, *batch_shape = inputs.shape
        outputs = (inputs - self._shift) / self._scale
        logabsdet = -self._batch_logabsdet(batch_shape).expand(num_batches)

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


AffineScalarTransform = AffineTransform

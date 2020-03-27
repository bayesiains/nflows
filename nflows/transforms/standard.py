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

    # XXX memoize?
    def _batch_logabsdet(self, batch_shape: torch.Size):
        """Return log abs det with input batch shape."""
        scale_numel = int(np.prod(self._scale.shape))
        if scale_numel > 1:
            return self._log_scale.expand(batch_shape).sum()
        else:
            # optimise for scalar scale: expand = numel equal scalars
            # also needed to match test at eps < 1E-5
            batch_numel = int(np.prod(batch_shape))
            return self._log_scale * batch_numel

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

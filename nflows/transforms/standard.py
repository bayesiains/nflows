"""Implementations of some standard transforms."""

from typing import Iterable, Optional, Tuple, Union
import warnings

import torch
from torch import Tensor

from nflows.transforms.base import Transform


class IdentityTransform(Transform):
    """Transform that leaves input unchanged."""

    def forward(self, inputs: Tensor, context=Optional[Tensor]):
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)
        return inputs, logabsdet

    def inverse(self, inputs: Tensor, context=Optional[Tensor]):
        return self(inputs, context)


class PointwiseAffineTransform(Transform):
    """Forward transform X = X * scale + shift."""

    def __init__(
        self, shift: Union[Tensor, float] = 0.0, scale: Union[Tensor, float] = 1.0,
    ):
        super().__init__()
        shift, scale = map(torch.as_tensor, (shift, scale))

        if (scale == 0.0).any():
            raise ValueError("Scale must be non-zero.")

        self.register_buffer("_shift", shift)
        self.register_buffer("_scale", scale)

    @property
    def _log_abs_scale(self) -> Tensor:
        return torch.log(torch.abs(self._scale))

    # XXX Memoize result on first run?
    def _batch_logabsdet(self, batch_shape: Iterable[int]) -> Tensor:
        """Return log abs det with input batch shape."""

        if self._log_abs_scale.numel() > 1:
            return self._log_abs_scale.expand(batch_shape).sum()
        else:
            # When log_abs_scale is a scalar, we use n*log_abs_scale, which is more
            # numerically accurate than \sum_1^n log_abs_scale.
            return self._log_abs_scale * torch.Size(batch_shape).numel()

    def forward(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor]:
        batch_size, *batch_shape = inputs.size()

        # RuntimeError here means shift/scale not broadcastable to input.
        outputs = inputs * self._scale + self._shift
        logabsdet = self._batch_logabsdet(batch_shape).expand(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor]:
        batch_size, *batch_shape = inputs.size()
        outputs = (inputs - self._shift) / self._scale
        logabsdet = -self._batch_logabsdet(batch_shape).expand(batch_size)

        return outputs, logabsdet


class AffineTransform(PointwiseAffineTransform):
    def __init__(
        self, shift: Union[Tensor, float] = 0.0, scale: Union[Tensor, float] = 1.0,
    ):

        warnings.warn("Use PointwiseAffineTransform", DeprecationWarning)

        if shift is None:
            shift = 0.0
            warnings.warn(f"`shift=None` deprecated; default is {shift}")

        if scale is None:
            scale = 1.0
            warnings.warn(f"`scale=None` deprecated; default is {scale}.")

        super().__init__(shift, scale)


# Alias for backward compatibility.
AffineScalarTransform = AffineTransform

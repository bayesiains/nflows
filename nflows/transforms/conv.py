from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.utils import torchutils


class OneByOneConvolution(LULinear):
    """An invertible 1x1 convolution with a fixed permutation, as introduced in the Glow paper.

    Reference:
    > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
    """

    def __init__(self, num_channels, using_cache=False, identity_init=True):
        super().__init__(num_channels, using_cache, identity_init)
        self.permutation = RandomPermutation(num_channels, dim=1)

    def _lu_forward_inverse(self, inputs, inverse=False):
        b, c, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(b * h * w, c)

        if inverse:
            outputs, logabsdet = super().inverse(inputs)
        else:
            outputs, logabsdet = super().forward(inputs)

        outputs = outputs.reshape(b, h, w, c).permute(0, 3, 1, 2)
        logabsdet = logabsdet.reshape(b, h, w)

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Inputs must be a 4D tensor.")

        inputs, _ = self.permutation(inputs)

        return self._lu_forward_inverse(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Inputs must be a 4D tensor.")

        outputs, logabsdet = self._lu_forward_inverse(inputs, inverse=True)

        outputs, _ = self.permutation.inverse(outputs)

        return outputs, logabsdet

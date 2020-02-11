"""Implementations of autoregressive flows."""

from torch.nn import functional as F

from pyknos import distributions
from pyknos import flows
from pyknos import transforms


class MaskedAutoregressiveFlow(flows.Flow):
    """An autoregressive flow that uses affine transforms with masking.

    Reference:
    > G. Papamakarios et al., Masked Autoregressive Flow for Density Estimation,
    > Advances in Neural Information Processing Systems, 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_random_permutations:
            permutation_constructor = transforms.RandomPermutation
        else:
            permutation_constructor = transforms.ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                transforms.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(transforms.BatchNorm(features))

        super().__init__(
            transform=transforms.CompositeTransform(layers),
            distribution=distributions.StandardNormal([features]),
        )

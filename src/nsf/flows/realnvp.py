"""Implementations of Real NVP."""

import torch
from torch.nn import functional as F

from nsf import distributions
from nsf import flows
from nsf import transforms
from nn_ import nets as nn_


class SimpleRealNVP(flows.Flow):
    """An simplified version of Real NVP for 1-dim inputs.

    This implementation uses 1-dim checkerboard masking but doesn't use multi-scaling.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = transforms.AdditiveCouplingTransform
        else:
            coupling_constructor = transforms.AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nn_.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(transforms.BatchNorm(features=features))

        super().__init__(
            transform=transforms.CompositeTransform(layers),
            distribution=distributions.StandardNormal([features]),
        )

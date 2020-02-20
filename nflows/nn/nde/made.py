"""Implementation of MADE."""
# TODO: should be moved to module nets.

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import distributions, nn
from torch.nn import functional as F
from torch.nn import init

from nflows.utils import torchutils


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(
        self,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        bias=True,
    ):
        super().__init__(
            in_features=len(in_degrees), out_features=out_features, bias=bias
        )
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output,
        )
        self.register_buffer("mask", mask)
        self.register_buffer("degrees", degrees)

    @classmethod
    def _get_mask_and_degrees(
        cls, in_degrees, out_features, autoregressive_features, random_mask, is_output
    ):
        if is_output:
            out_degrees = torchutils.tile(
                _get_input_degrees(autoregressive_features),
                out_features // autoregressive_features,
            )
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long,
                )
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=False,
    ):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        if random_mask:
            raise ValueError("Masked residual block can't be used with random masks.")
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError(
                "In a masked residual block, the output degrees can't be"
                " less than the corresponding input degrees."
            )

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if context is not None:
            temps += self.context_layer(context)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    zero_initialization=True,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
        )

    def forward(self, inputs, context=None):
        temps = self.initial_layer(inputs)
        if context is not None:
            temps += self.context_layer(context)
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs


class MixtureOfGaussiansMADE(MADE):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        num_mixture_components=5,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        epsilon=1e-2,
        custom_initialization=True,
    ):

        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")

        super().__init__(
            features,
            hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=3 * num_mixture_components,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        self.num_mixture_components = num_mixture_components
        self.features = features
        self.hidden_features = hidden_features
        self.epsilon = epsilon

        if custom_initialization:
            self._initialize()

    def forward(self, inputs, context=None):
        return super().forward(inputs, context=context)

    def log_prob(self, inputs, context=None):
        outputs = self.forward(inputs, context=context)
        outputs = outputs.reshape(*inputs.shape, self.num_mixture_components, 3)

        logits, means, unconstrained_stds = (
            outputs[..., 0],
            outputs[..., 1],
            outputs[..., 2],
        )
        log_mixture_coefficients = torch.log_softmax(logits, dim=-1)
        stds = F.softplus(unconstrained_stds) + self.epsilon

        log_prob = torch.sum(
            torch.logsumexp(
                log_mixture_coefficients
                - 0.5
                * (
                    np.log(2 * np.pi)
                    + 2 * torch.log(stds)
                    + ((inputs[..., None] - means) / stds) ** 2
                ),
                dim=-1,
            ),
            dim=-1,
        )
        return log_prob

    def sample(self, num_samples, context=None):

        if context is not None:
            context = torchutils.repeat_rows(context, num_samples)

        with torch.no_grad():

            samples = torch.zeros(context.shape[0], self.features)

            for feature in range(self.features):
                outputs = self.forward(samples, context)
                outputs = outputs.reshape(
                    *samples.shape, self.num_mixture_components, 3
                )

                logits, means, unconstrained_stds = (
                    outputs[:, feature, :, 0],
                    outputs[:, feature, :, 1],
                    outputs[:, feature, :, 2],
                )
                logits = torch.log_softmax(logits, dim=-1)
                stds = F.softplus(unconstrained_stds) + self.epsilon

                component_distribution = distributions.Categorical(logits=logits)
                components = component_distribution.sample((1,)).reshape(-1, 1)
                means, stds = (
                    means.gather(1, components).reshape(-1),
                    stds.gather(1, components).reshape(-1),
                )
                samples[:, feature] = (
                    means + torch.randn(context.shape[0]) * stds
                ).detach()

        return samples.reshape(-1, num_samples, self.features)

    def _initialize(self):
        # Initialize mixture coefficient logits to near zero so that mixture coefficients
        # are approximately uniform.
        self.final_layer.weight.data[::3, :] = self.epsilon * torch.randn(
            self.features * self.num_mixture_components, self.hidden_features
        )
        self.final_layer.bias.data[::3] = self.epsilon * torch.randn(
            self.features * self.num_mixture_components
        )

        # self.final_layer.weight.data[1::3, :] = self.epsilon * torch.randn(
        #     self.features * self.num_mixture_components, self.hidden_features
        # )
        # low, high = -7, 7
        # self.final_layer.bias.data[1::3] = (high - low) * torch.rand(
        #     self.features * self.num_mixture_components
        # ) + low

        # Initialize unconstrained standard deviations to the inverse of the softplus
        # at 1 so that they're near 1 at initialization.
        self.final_layer.weight.data[2::3] = self.epsilon * torch.randn(
            self.features * self.num_mixture_components, self.hidden_features
        )
        self.final_layer.bias.data[2::3] = torch.log(
            torch.exp(torch.Tensor([1 - self.epsilon])) - 1
        ) * torch.ones(
            self.features * self.num_mixture_components
        ) + self.epsilon * torch.randn(
            self.features * self.num_mixture_components
        )
        # self.final_layer.bias.data[2::3] = torch.log(
        #     torch.Tensor([1 - self.epsilon])
        # ) * torch.ones(
        #     self.features * self.num_mixture_components
        # ) + self.epsilon * torch.randn(
        #     self.features * self.num_mixture_components
        # )


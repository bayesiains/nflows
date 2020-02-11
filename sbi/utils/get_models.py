from torch import nn
from torch.nn import functional as F

from sbi.nn_ import nets as nn_

from nsf import distributions as distributions_, flows, transforms
from sbi.nn_.nde import MixtureOfGaussiansMADE, MultivariateGaussianMDN

from .torchutils import create_alternating_binary_mask


def get_neural_posterior(model, parameter_dim, observation_dim, simulator):

    # Everything is a flow because we need to normalize parameters based on prior.

    mean, std = simulator.normalization_parameters
    normalizing_transform = transforms.AffineTransform(shift=-mean / std, scale=1 / std)

    if model == "mdn":
        hidden_features = 50
        neural_posterior = MultivariateGaussianMDN(
            features=parameter_dim,
            context_features=observation_dim,
            hidden_features=hidden_features,
            hidden_net=nn.Sequential(
                nn.Linear(observation_dim, hidden_features),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
            ),
            num_components=20,
            custom_initialization=True,
        )

    elif model == "made":
        num_mixture_components = 5
        transform = normalizing_transform
        distribution = distributions_.MADEMoG(
            features=parameter_dim,
            hidden_features=50,
            context_features=observation_dim,
            num_blocks=2,
            num_mixture_components=num_mixture_components,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            custom_initialization=True,
        )
        neural_posterior = flows.Flow(transform, distribution)

    elif model == "maf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=parameter_dim,
                            hidden_features=50,
                            context_features=observation_dim,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=F.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=parameter_dim),
                    ]
                )
                for _ in range(5)
            ]
        )

        transform = transforms.CompositeTransform([normalizing_transform, transform,])

        distribution = distributions_.StandardNormal((parameter_dim,))
        neural_posterior = flows.Flow(transform, distribution)

    elif model == "nsf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.PiecewiseRationalQuadraticCouplingTransform(
                            mask=create_alternating_binary_mask(
                                features=parameter_dim, even=(i % 2 == 0)
                            ),
                            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=50,
                                context_features=observation_dim,
                                num_blocks=2,
                                activation=F.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                            ),
                            num_bins=10,
                            tails="linear",
                            tail_bound=3.0,
                            apply_unconditional_transform=False,
                        ),
                        transforms.LULinear(parameter_dim, identity_init=True),
                    ]
                )
                for i in range(5)
            ]
        )

        distribution = distributions_.StandardNormal((parameter_dim,))
        neural_posterior = flows.Flow(transform, distribution)

    else:
        raise ValueError

    return neural_posterior


def get_classifier(model, parameter_dim, observation_dim):

    if model == "linear":
        classifier = nn.Linear(parameter_dim + observation_dim, 1)

    elif model == "mlp":
        hidden_dim = 50
        classifier = nn.Sequential(
            nn.Linear(parameter_dim + observation_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    elif model == "resnet":
        classifier = nn_.ResidualNet(
            in_features=parameter_dim + observation_dim,
            out_features=1,
            hidden_features=50,
            context_features=None,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    else:
        raise ValueError(f"'model' must be one of ['linear', 'mlp', 'resnet'].")

    return classifier


def get_neural_likelihood(model, parameter_dim, observation_dim):

    if model == "mdn":
        hidden_features = 50
        neural_likelihood = MultivariateGaussianMDN(
            features=observation_dim,
            context_features=parameter_dim,
            hidden_features=hidden_features,
            hidden_net=nn.Sequential(
                nn.Linear(parameter_dim, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(hidden_features, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
            ),
            num_components=20,
            custom_initialization=True,
        )

    elif model == "made":
        neural_likelihood = MixtureOfGaussiansMADE(
            features=observation_dim,
            hidden_features=50,
            context_features=parameter_dim,
            num_blocks=4,
            num_mixture_components=10,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            use_batch_norm=True,
            dropout_probability=0.0,
            custom_initialization=True,
        )

    elif model == "maf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=observation_dim,
                            hidden_features=50,
                            context_features=parameter_dim,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=F.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=observation_dim),
                    ]
                )
                for _ in range(5)
            ]
        )
        distribution = distributions_.StandardNormal((observation_dim,))
        neural_likelihood = flows.Flow(transform, distribution)

    elif model == "nsf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.PiecewiseRationalQuadraticCouplingTransform(
                            mask=create_alternating_binary_mask(
                                features=observation_dim, even=(i % 2 == 0)
                            ),
                            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=50,
                                context_features=parameter_dim,
                                num_blocks=2,
                                activation=F.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                            ),
                            num_bins=10,
                            tails="linear",
                            tail_bound=3.0,
                            apply_unconditional_transform=False,
                        ),
                        transforms.LULinear(observation_dim, identity_init=True),
                    ]
                )
                for i in range(5)
            ]
        )
        distribution = distributions_.StandardNormal((observation_dim,))
        neural_likelihood = flows.Flow(transform, distribution)

    else:
        raise ValueError

    return neural_likelihood

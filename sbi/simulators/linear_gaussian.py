import torch

from sbi.simulators.simulator import Simulator


class LinearGaussianSimulator(Simulator):
    """
    Implemenation of linear Gaussian simulator.
    Observations are generating by adding diagonal Gaussian noise of a specified variance
    to given parameters.
    """

    def __init__(self, dim=2, std=1, use_zero_ground_truth=True):
        """
        :param dim: int
            Dimension of the parameters and observations.
        :param std: positive float
            Standard deviation of diagonal Gaussian shared across dimensions.
        :param use_zero_ground_truth: bool
            Use the zero vector as a ground truth observation.
        """
        super().__init__()
        self._std = std
        self._dim = dim
        self._use_zero_ground_truth = use_zero_ground_truth

        # Generate ground truth samples to return if requested.
        self._ground_truth_samples = self._sample_ground_truth_posterior(
            num_samples=10000
        )

    def simulate(self, parameters):
        """
        Generates noisy observations of the given batch of parameters.

        :param parameters: torch.Tensor
            Batch of parameters.
        :return: torch.Tensor
            Parameters with diagonal Gaussian noise with shared variance across dimensions
            added.
        """
        if parameters.ndim == 1:
            parameters = parameters[None, :]
        return parameters + self._std * torch.randn_like(parameters)

    def get_ground_truth_parameters(self):
        """
        True parameters always the zero vector.

        :return: torch.Tensor
            Ground truth parameters.
        """
        return torch.zeros(self._dim)

    def get_ground_truth_observation(self):
        """
        Ground truth observation is either the zero vector, or a noisy observation of the
        zero vector.

        :return: torch.Tensor
            Ground truth observation.
        """
        if self._use_zero_ground_truth:
            return torch.zeros(self._dim)
        else:
            return self._std * torch.randn(self._dim)

    def _sample_ground_truth_posterior(self, num_samples=1000):
        """
        Samples from ground truth posterior assuming prior is standard normal.

        :param num_samples: int
            Number of samples to draw.
        :return: torch.Tensor [num_samples, observation_dim]
            Batch of posterior samples.
        """
        mean = self.get_ground_truth_parameters()
        std = torch.sqrt(torch.Tensor([self._std ** 2 / (self._std ** 2 + 1)]))
        c = torch.Tensor([1 / (self._std ** 2 + 1)])
        return c * mean + std * torch.randn(num_samples, self._dim)

    def get_ground_truth_posterior_samples(self, num_samples=1000):
        """
        Returns first num_samples samples we have stored if there are enough,
        otherwise generates sufficiently many and returns those.

        :param num_samples: int
            Number of samples to generate.
        :return: torch.Tensor [batch_size, observation_dim]
            Batch of posterior samples.
        """
        if num_samples < self._ground_truth_samples.shape[0]:
            return self._ground_truth_samples[:num_samples]
        else:
            self._ground_truth_samples = self._sample_ground_truth_posterior(
                num_samples=num_samples
            )
            return self._ground_truth_samples

    @property
    def parameter_dim(self):
        return self._dim

    @property
    def observation_dim(self):
        return self._dim

    @property
    def name(self):
        return "linear-gaussian"

    @property
    def parameter_plotting_limits(self):
        return [-4, 4]

    @property
    def normalization_parameters(self):
        mean = torch.zeros(self._dim)
        std = torch.ones(self._dim)
        return mean, std

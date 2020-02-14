"""Basic definitions for the flows module."""

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution

    def _log_prob(self, inputs, context):
        noise, logabsdet = self._transform(inputs, context=context)
        log_prob = self._distribution.log_prob(noise, context=context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)

        samples, _ = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=context
        )

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=context)
        return noise

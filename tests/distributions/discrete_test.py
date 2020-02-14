"""Tests for discrete distributions."""

import unittest

import torch
import torchtestcase

from nflows.distributions import discrete


class ConditionalIndependentBernoulliTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 4]
        dist = discrete.ConditionalIndependentBernoulli(input_shape)
        inputs = torch.randn(batch_size, *input_shape)
        context = torch.randn(batch_size, *context_shape)
        log_prob = dist.log_prob(inputs, context=context)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())
        self.assert_tensor_less_equal(log_prob, 0.0)

    def test_sample(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 4]
        dist = discrete.ConditionalIndependentBernoulli(input_shape)
        context = torch.randn(context_size, *context_shape)
        samples = dist.sample(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + input_shape)
        )
        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())
        binary = (samples == 1.0) | (samples == 0.0)
        self.assertEqual(binary, torch.ones_like(binary))

    def test_sample_and_log_prob_with_context(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 4]

        dist = discrete.ConditionalIndependentBernoulli(input_shape)
        context = torch.randn(context_size, *context_shape)
        samples, log_prob = dist.sample_and_log_prob(num_samples, context=context)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)

        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([context_size, num_samples]))

        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())
        self.assert_tensor_less_equal(log_prob, 0.0)

        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())
        binary = (samples == 1.0) | (samples == 0.0)
        self.assertEqual(binary, torch.ones_like(binary))

    def test_mean(self):
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 4]
        dist = discrete.ConditionalIndependentBernoulli(input_shape)
        context = torch.randn(context_size, *context_shape)
        means = dist.mean(context=context)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, torch.Size([context_size] + input_shape))
        self.assertFalse(torch.isnan(means).any())
        self.assertFalse(torch.isinf(means).any())
        self.assert_tensor_greater_equal(means, 0.0)
        self.assert_tensor_less_equal(means, 1.0)


if __name__ == "__main__":
    unittest.main()

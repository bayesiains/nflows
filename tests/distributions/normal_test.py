"""Tests for Normal distributions."""

import unittest

import torch
import torchtestcase

from nflows.distributions import normal


class StandardNormalTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        dist = normal.StandardNormal(input_shape)
        inputs = torch.randn(batch_size, *input_shape)
        maybe_context = torch.randn(batch_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                log_prob = dist.log_prob(inputs, context=context)
                self.assertIsInstance(log_prob, torch.Tensor)
                self.assertEqual(log_prob.shape, torch.Size([batch_size]))
                self.assertFalse(torch.isnan(log_prob).any())
                self.assertFalse(torch.isinf(log_prob).any())

    def test_sample(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        dist = normal.StandardNormal(input_shape)
        maybe_context = torch.randn(context_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                samples = dist.sample(num_samples, context=context)
                self.assertIsInstance(samples, torch.Tensor)
                self.assertFalse(torch.isnan(samples).any())
                self.assertFalse(torch.isinf(samples).any())
                if context is None:
                    self.assertEqual(
                        samples.shape, torch.Size([num_samples] + input_shape)
                    )
                else:
                    self.assertEqual(
                        samples.shape,
                        torch.Size([context_size, num_samples] + input_shape),
                    )

    def test_sample_and_log_prob(self):
        num_samples = 10
        input_shape = [2, 3, 4]
        dist = normal.StandardNormal(input_shape)
        samples, log_prob_1 = dist.sample_and_log_prob(num_samples)
        log_prob_2 = dist.log_prob(samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob_1, torch.Tensor)
        self.assertIsInstance(log_prob_2, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples] + input_shape))
        self.assertEqual(log_prob_1.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_2.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_1, log_prob_2)

    def test_sample_and_log_prob_with_context(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        dist = normal.StandardNormal(input_shape)
        context = torch.randn(context_size, *context_shape)
        samples, log_prob = dist.sample_and_log_prob(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([context_size, num_samples]))

    def test_mean(self):
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        dist = normal.StandardNormal(input_shape)
        maybe_context = torch.randn(context_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                means = dist.mean(context=context)
                self.assertIsInstance(means, torch.Tensor)
                self.assertFalse(torch.isnan(means).any())
                self.assertFalse(torch.isinf(means).any())
                self.assertEqual(means, torch.zeros_like(means))
                if context is None:
                    self.assertEqual(means.shape, torch.Size(input_shape))
                else:
                    self.assertEqual(
                        means.shape, torch.Size([context_size] + input_shape)
                    )


class ConditionalDiagonalNormalTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 8]
        dist = normal.ConditionalDiagonalNormal(input_shape)
        inputs = torch.randn(batch_size, *input_shape)
        context = torch.randn(batch_size, *context_shape)
        log_prob = dist.log_prob(inputs, context=context)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())

    def test_sample(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 8]
        dist = normal.ConditionalDiagonalNormal(input_shape)
        context = torch.randn(context_size, *context_shape)
        samples = dist.sample(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + input_shape)
        )
        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())

    def test_sample_and_log_prob_with_context(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 8]
        dist = normal.ConditionalDiagonalNormal(input_shape)
        context = torch.randn(context_size, *context_shape)
        samples, log_prob = dist.sample_and_log_prob(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([context_size, num_samples]))

    def test_mean(self):
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [2, 3, 8]
        dist = normal.ConditionalDiagonalNormal(input_shape)
        context = torch.randn(context_size, *context_shape)
        means = dist.mean(context=context)
        self.assertIsInstance(means, torch.Tensor)
        self.assertFalse(torch.isnan(means).any())
        self.assertFalse(torch.isinf(means).any())
        self.assertEqual(means.shape, torch.Size([context_size] + input_shape))


if __name__ == "__main__":
    unittest.main()

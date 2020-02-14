"""Tests for the basic flow definitions."""

import unittest

import torch
import torchtestcase

from nflows.distributions.normal import StandardNormal
from nflows.flows import base
from nflows.transforms.standard import AffineScalarTransform


class FlowTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        flow = base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(input_shape),
        )
        inputs = torch.randn(batch_size, *input_shape)
        maybe_context = torch.randn(batch_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                log_prob = flow.log_prob(inputs, context=context)
                self.assertIsInstance(log_prob, torch.Tensor)
                self.assertEqual(log_prob.shape, torch.Size([batch_size]))

    def test_sample(self):
        num_samples = 10
        context_size = 20
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        flow = base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(input_shape),
        )
        maybe_context = torch.randn(context_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                samples = flow.sample(num_samples, context=context)
                self.assertIsInstance(samples, torch.Tensor)
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
        flow = base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(input_shape),
        )
        samples, log_prob_1 = flow.sample_and_log_prob(num_samples)
        log_prob_2 = flow.log_prob(samples)
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
        flow = base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(input_shape),
        )
        context = torch.randn(context_size, *context_shape)
        samples, log_prob = flow.sample_and_log_prob(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([context_size, num_samples]))

    def test_transform_to_noise(self):
        batch_size = 10
        context_size = 20
        shape = [2, 3, 4]
        context_shape = [5, 6]
        flow = base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(shape),
        )
        inputs = torch.randn(batch_size, *shape)
        maybe_context = torch.randn(context_size, *context_shape)
        for context in [None, maybe_context]:
            with self.subTest(context=context):
                noise = flow.transform_to_noise(inputs, context=context)
                self.assertIsInstance(noise, torch.Tensor)
                self.assertEqual(noise.shape, torch.Size([batch_size] + shape))


if __name__ == "__main__":
    unittest.main()

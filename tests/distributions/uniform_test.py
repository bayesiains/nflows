"""Tests for BoxUniform distribution."""

import unittest

import torch
import torchtestcase

from nflows.distributions import uniform

class BoxUniformTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):        
        batch_size = 10
        input_shape = [2, 3, 4]
        context_shape = [5, 6]
        low = -torch.rand(input_shape)
        high = torch.rand(input_shape)
        dist = uniform.BoxUniform(low, high)
        inputs = torch.rand(batch_size, *input_shape)
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
        low = -torch.rand(input_shape)
        high = torch.rand(input_shape)
        dist = uniform.BoxUniform(low, high)
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
        low = -torch.rand(input_shape)
        high = torch.rand(input_shape)
        dist = uniform.BoxUniform(low, high)
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
        low = -torch.rand(input_shape)
        high = torch.rand(input_shape)
        dist = uniform.BoxUniform(low, high)
        context = torch.randn(context_size, *context_shape)
        samples, log_prob = dist.sample_and_log_prob(num_samples, context=context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(
            samples.shape, torch.Size([context_size, num_samples] + input_shape)
        )
        self.assertEqual(log_prob.shape, torch.Size([context_size, num_samples]))

if __name__ == "__main__":
    unittest.main()
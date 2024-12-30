"""Tests for Real NVP."""

import unittest

import torch
import torchtestcase

from nflows.flows import realnvp


class SimpleRealNVPTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        features = 20
        flow = realnvp.SimpleRealNVP(
            features=features, hidden_features=30, num_layers=5, num_blocks_per_layer=2
        )
        inputs = torch.randn(batch_size, features)
        log_prob = flow.log_prob(inputs)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))

    def test_sample(self):
        num_samples = 10
        features = 20
        flow = realnvp.SimpleRealNVP(
            features=features, hidden_features=30, num_layers=5, num_blocks_per_layer=2
        )
        samples = flow.sample(num_samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples, features]))

    def test_conditional_log_prob(self):
        batch_size = 10
        features = 20
        context_features = 5
        flow = realnvp.SimpleRealNVP(
            features=features, hidden_features=30, context_features=context_features
        )
        inputs = torch.randn(batch_size, features)
        context = torch.randn(batch_size, context_features)
        log_prob = flow.log_prob(inputs, context)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))

    def test_conditional_sample(self):
        num_samples = 10

        batch_size = 10
        features = 20
        context_features = 5
        flow = realnvp.SimpleRealNVP(
            features=features, hidden_features=30, context_features=context_features
        )
        context = torch.randn(batch_size, context_features)
        samples = flow.sample(num_samples, context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([batch_size, num_samples, features]))


if __name__ == "__main__":
    unittest.main()

"""Tests for linear transforms."""

import unittest
from unittest.mock import MagicMock

import torch

from nflows.transforms import linear
from nflows.transforms.linear import Linear
from nflows.utils import torchutils
from tests.transforms.transform_test import TransformTest


class LinearTest(TransformTest):
    def setUp(self):
        features = 5
        batch_size = 10

        weight = torch.randn(features, features)
        inverse = torch.randn(features, features)
        logabsdet = torch.randn(1)
        self.transform = Linear(features)
        self.transform.bias.data = torch.randn(features)  # Just so bias isn't zero.

        self.inputs = torch.randn(batch_size, features)
        self.outputs_fwd = self.inputs @ weight.t() + self.transform.bias
        self.outputs_inv = (self.inputs - self.transform.bias) @ inverse.t()
        self.logabsdet_fwd = logabsdet * torch.ones(batch_size)
        self.logabsdet_inv = (-logabsdet) * torch.ones(batch_size)

        # Mocks for abstract methods.
        self.transform.forward_no_cache = MagicMock(
            return_value=(self.outputs_fwd, self.logabsdet_fwd)
        )
        self.transform.inverse_no_cache = MagicMock(
            return_value=(self.outputs_inv, self.logabsdet_inv)
        )
        self.transform.weight = MagicMock(return_value=weight)
        self.transform.weight_inverse = MagicMock(return_value=inverse)
        self.transform.logabsdet = MagicMock(return_value=logabsdet)

    def test_forward_default(self):
        outputs, logabsdet = self.transform(self.inputs)

        self.transform.forward_no_cache.assert_called_with(self.inputs)
        self.assertEqual(outputs, self.outputs_fwd)
        self.assertEqual(logabsdet, self.logabsdet_fwd)

        # Cache shouldn't be computed.
        self.assertFalse(self.transform.weight.called)
        self.assertFalse(self.transform.logabsdet.called)

    def test_inverse_default(self):
        outputs, logabsdet = self.transform.inverse(self.inputs)

        self.transform.inverse_no_cache.assert_called_with(self.inputs)
        self.assertEqual(outputs, self.outputs_inv)
        self.assertEqual(logabsdet, self.logabsdet_inv)

        # Cache shouldn't be computed.
        self.assertFalse(self.transform.weight_inverse.called)
        self.assertFalse(self.transform.logabsdet.called)

    def test_forward_cached(self):
        self.transform.eval()
        self.transform.use_cache()

        outputs, logabsdet = self.transform(self.inputs)
        self.assertTrue(self.transform.weight.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.assertEqual(outputs, self.outputs_fwd)
        self.assertEqual(logabsdet, self.logabsdet_fwd)

    def test_inverse_cached(self):
        self.transform.eval()
        self.transform.use_cache()

        outputs, logabsdet = self.transform.inverse(self.inputs)
        self.assertTrue(self.transform.weight_inverse.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.assertEqual(outputs, self.outputs_inv)
        self.assertEqual(logabsdet, self.logabsdet_inv)

    def test_forward_cache_is_used(self):
        self.transform.eval()
        self.transform.use_cache()

        self.transform(self.inputs)
        self.assertTrue(self.transform.weight.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.transform.weight.reset_mock()
        self.transform.logabsdet.reset_mock()

        outputs, logabsdet = self.transform(self.inputs)
        # Cached values should be used.
        self.assertFalse(self.transform.weight.called)
        self.assertFalse(self.transform.logabsdet.called)
        self.assertEqual(outputs, self.outputs_fwd)
        self.assertEqual(logabsdet, self.logabsdet_fwd)

    def test_inverse_cache_is_used(self):
        self.transform.eval()
        self.transform.use_cache()

        self.transform.inverse(self.inputs)
        self.assertTrue(self.transform.weight_inverse.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.transform.weight_inverse.reset_mock()
        self.transform.logabsdet.reset_mock()

        outputs, logabsdet = self.transform.inverse(self.inputs)
        # Cached values should be used.
        self.assertFalse(self.transform.weight_inverse.called)
        self.assertFalse(self.transform.logabsdet.called)
        self.assertEqual(outputs, self.outputs_inv)
        self.assertEqual(logabsdet, self.logabsdet_inv)

    def test_forward_cache_not_used_while_training(self):
        self.transform.train()
        self.transform.use_cache()

        outputs, logabsdet = self.transform(self.inputs)
        self.transform.forward_no_cache.assert_called_with(self.inputs)
        self.assertEqual(outputs, self.outputs_fwd)
        self.assertEqual(logabsdet, self.logabsdet_fwd)

        # Cache shouldn't be computed.
        self.assertFalse(self.transform.weight.called)
        self.assertFalse(self.transform.logabsdet.called)

    def test_inverse_cache_not_used_while_training(self):
        self.transform.train()
        self.transform.use_cache()

        outputs, logabsdet = self.transform.inverse(self.inputs)
        self.transform.inverse_no_cache.assert_called_with(self.inputs)
        self.assertEqual(outputs, self.outputs_inv)
        self.assertEqual(logabsdet, self.logabsdet_inv)

        # Cache shouldn't be computed.
        self.assertFalse(self.transform.weight_inverse.called)
        self.assertFalse(self.transform.logabsdet.called)

    def test_forward_train_invalidates_cache(self):
        self.transform.eval()
        self.transform.use_cache()

        self.transform(self.inputs)
        self.assertTrue(self.transform.weight.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.transform.weight.reset_mock()
        self.transform.logabsdet.reset_mock()

        self.transform.train()  # Cache should be invalidated here.
        self.assertTrue(
            self.transform.using_cache
        )  # Using cache should still be enabled.
        self.transform.eval()

        outputs, logabsdet = self.transform(self.inputs)
        # Values should be recomputed.
        self.assertTrue(self.transform.weight.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.assertEqual(outputs, self.outputs_fwd)
        self.assertEqual(logabsdet, self.logabsdet_fwd)

    def test_inverse_train_invalidates_cache(self):
        self.transform.eval()
        self.transform.use_cache()

        self.transform.inverse(self.inputs)
        self.assertTrue(self.transform.weight_inverse.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.transform.weight_inverse.reset_mock()
        self.transform.logabsdet.reset_mock()

        self.transform.train()  # Cache should be disabled and invalidated here.
        self.assertTrue(
            self.transform.using_cache
        )  # Using cache should still be enabled.
        self.transform.eval()

        outputs, logabsdet = self.transform.inverse(self.inputs)
        # Values should be recomputed.
        self.assertTrue(self.transform.weight_inverse.called)
        self.assertTrue(self.transform.logabsdet.called)
        self.assertEqual(outputs, self.outputs_inv)
        self.assertEqual(logabsdet, self.logabsdet_inv)


class NaiveLinearTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.transform = linear.NaiveLinear(features=self.features)

        self.weight = self.transform._weight
        self.weight_inverse = torch.inverse(self.weight)
        self.logabsdet = torchutils.logabsdet(self.weight)

        self.eps = 1e-5

    def test_forward_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        outputs, logabsdet = self.transform.forward_no_cache(inputs)

        outputs_ref = inputs @ self.weight.t() + self.transform.bias
        logabsdet_ref = torch.full([batch_size], self.logabsdet.item())

        self.assert_tensor_is_good(outputs, [batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [batch_size])

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse_no_cache(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        outputs, logabsdet = self.transform.inverse_no_cache(inputs)

        outputs_ref = (inputs - self.transform.bias) @ self.weight_inverse.t()
        logabsdet_ref = torch.full([batch_size], -self.logabsdet.item())

        self.assert_tensor_is_good(outputs, [batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [batch_size])

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_weight(self):
        weight = self.transform.weight()
        self.assert_tensor_is_good(weight, [self.features, self.features])
        self.assertEqual(weight, self.weight)

    def test_weight_inverse(self):
        weight_inverse = self.transform.weight_inverse()
        self.assert_tensor_is_good(weight_inverse, [self.features, self.features])
        self.assertEqual(weight_inverse, self.weight_inverse)

    def test_logabsdet(self):
        logabsdet = self.transform.logabsdet()
        self.assert_tensor_is_good(logabsdet, [])
        self.assertEqual(logabsdet, self.logabsdet)

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        inputs = torch.randn(batch_size, self.features)
        self.assert_forward_inverse_are_consistent(self.transform, inputs)


if __name__ == "__main__":
    unittest.main()

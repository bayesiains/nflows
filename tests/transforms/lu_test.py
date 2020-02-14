"""Tests for the LU linear transforms."""

import unittest

import torch

from nflows.transforms import lu
from nflows.utils import torchutils
from tests.transforms.transform_test import TransformTest


class LULinearTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.transform = lu.LULinear(features=self.features)

        lower, upper = self.transform._create_lower_upper()
        self.weight = lower @ upper
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

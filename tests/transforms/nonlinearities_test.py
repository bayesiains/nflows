"""Tests for the invertible non-linearities."""

import unittest

import torch

from nflows.transforms import nonlinearities as nl
from nflows.transforms import standard
from nflows.transforms.base import InputOutsideDomain
from tests.transforms.transform_test import TransformTest


class TanhTest(TransformTest):
    def test_raises_domain_exception(self):
        shape = [2, 3, 4]
        transform = nl.Tanh()
        for value in [-2.0, -1.0, 1.0, 2.0]:
            with self.assertRaises(InputOutsideDomain):
                inputs = torch.full(shape, value)
                transform.inverse(inputs)


class TestPiecewiseCDF(TransformTest):
    def setUp(self):
        self.shape = [2, 3, 4]
        self.batch_size = 10
        self.transforms = [
            nl.PiecewiseLinearCDF(self.shape),
            nl.PiecewiseQuadraticCDF(self.shape),
            nl.PiecewiseCubicCDF(self.shape),
            nl.PiecewiseRationalQuadraticCDF(self.shape),
        ]

    def test_raises_domain_exception(self):
        for transform in self.transforms:
            with self.subTest(transform=transform):
                for value in [-1.0, -0.1, 1.1, 2.0]:
                    with self.assertRaises(InputOutsideDomain):
                        inputs = torch.full([self.batch_size] + self.shape, value)
                        transform.forward(inputs)

    def test_zeros_to_zeros(self):
        for transform in self.transforms:
            with self.subTest(transform=transform):
                inputs = torch.zeros(self.batch_size, *self.shape)
                outputs, _ = transform(inputs)
                self.eps = 1e-5
                self.assertEqual(outputs, inputs)

    def test_ones_to_ones(self):
        for transform in self.transforms:
            with self.subTest(transform=transform):
                inputs = torch.ones(self.batch_size, *self.shape)
                outputs, _ = transform(inputs)
                self.eps = 1e-5
                self.assertEqual(outputs, inputs)

    def test_forward_inverse_are_consistent(self):
        for transform in self.transforms:
            with self.subTest(transform=transform):
                inputs = torch.rand(self.batch_size, *self.shape)
                self.eps = 1e-4
                self.assert_forward_inverse_are_consistent(transform, inputs)


class TestUnconstrainedPiecewiseCDF(TransformTest):
    def test_forward_inverse_are_consistent(self):
        shape = [2, 3, 4]
        batch_size = 10
        transforms = [
            nl.PiecewiseLinearCDF(shape, tails="linear"),
            nl.PiecewiseQuadraticCDF(shape, tails="linear"),
            nl.PiecewiseCubicCDF(shape, tails="linear"),
            nl.PiecewiseRationalQuadraticCDF(shape, tails="linear"),
        ]

        for transform in transforms:
            with self.subTest(transform=transform):
                inputs = 3 * torch.randn(batch_size, *shape)
                self.eps = 1e-4
                self.assert_forward_inverse_are_consistent(transform, inputs)


class LogitTest(TransformTest):
    def test_forward_zero_and_one(self):
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.cat(
            [torch.zeros(batch_size // 2, *shape), torch.ones(batch_size // 2, *shape)]
        )

        transform = nl.Logit()
        outputs, logabsdet = transform(inputs)

        self.assert_tensor_is_good(outputs)
        self.assert_tensor_is_good(logabsdet)


class NonlinearitiesTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.rand(batch_size, *shape)
        transforms = [
            nl.Tanh(),
            nl.LogTanh(),
            nl.LeakyReLU(),
            nl.Sigmoid(),
            nl.Logit(),
            nl.CompositeCDFTransform(nl.Sigmoid(), standard.IdentityTransform()),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_inverse(self):
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.rand(batch_size, *shape)
        transforms = [
            nl.Tanh(),
            nl.LogTanh(),
            nl.LeakyReLU(),
            nl.Sigmoid(),
            nl.Logit(),
            nl.CompositeCDFTransform(nl.Sigmoid(), standard.IdentityTransform()),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        shape = [5, 10, 15]
        inputs = torch.rand(batch_size, *shape)
        transforms = [
            nl.Tanh(),
            nl.LogTanh(),
            nl.LeakyReLU(),
            nl.Sigmoid(),
            nl.Logit(),
            nl.CompositeCDFTransform(nl.Sigmoid(), standard.IdentityTransform()),
        ]
        self.eps = 1e-3
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()

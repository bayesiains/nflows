"""Tests for the autoregressive transforms."""

import unittest

import torch

from nflows.transforms import autoregressive
from tests.transforms.transform_test import TransformTest


class MaskedAffineAutoregressiveTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        features = 20
        inputs = torch.randn(batch_size, features)
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                transform = autoregressive.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_inverse(self):
        batch_size = 10
        features = 20
        inputs = torch.randn(batch_size, features)
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                transform = autoregressive.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.randn(batch_size, features)
        self.eps = 1e-6
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                transform = autoregressive.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseLinearAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3

        transform = autoregressive.MaskedPiecewiseLinearAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseQuadraticAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-4

        transform = autoregressive.MaskedPiecewiseQuadraticAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedUMNNAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-4

        transform = autoregressive.MaskedUMNNAutoregressiveTransform(
            cond_size=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseCubicAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3

        transform = autoregressive.MaskedPiecewiseCubicAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()

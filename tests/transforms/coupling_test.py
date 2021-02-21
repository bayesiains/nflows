"""Tests for the coupling Transforms."""

import unittest

import torch

from nflows.nn import nets
from nflows.transforms import coupling
from nflows.utils import torchutils
from tests.transforms.transform_test import TransformTest


def create_coupling_transform(cls, shape, **kwargs):
    if len(shape) == 1:

        def create_net(in_features, out_features):
            return nets.ResidualNet(
                in_features, out_features, hidden_features=30, num_blocks=5
            )

    else:

        def create_net(in_channels, out_channels):
            # return nets.Conv2d(in_channels, out_channels, kernel_size=1)
            return nets.ConvResidualNet(
                in_channels=in_channels, out_channels=out_channels, hidden_channels=16
            )

    mask = torchutils.create_mid_split_binary_mask(shape[0])

    return cls(mask=mask, transform_net_create_fn=create_net, **kwargs), mask


batch_size = 10


class AffineCouplingTransformTest(TransformTest):
    shapes = [[20], [2, 4, 4]]

    def test_forward(self):
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.AffineCouplingTransform, shape
            )
            outputs, logabsdet = transform(inputs)
            with self.subTest(shape=shape):
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...])

    def test_inverse(self):
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.AffineCouplingTransform, shape
            )
            outputs, logabsdet = transform(inputs)
            with self.subTest(shape=shape):
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...])

    def test_forward_inverse_are_consistent(self):
        self.eps = 1e-6
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.AffineCouplingTransform, shape
            )
            with self.subTest(shape=shape):
                self.assert_forward_inverse_are_consistent(transform, inputs)


class AdditiveTransformTest(TransformTest):
    shapes = [[20], [2, 4, 4]]

    def test_forward(self):
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.AdditiveCouplingTransform, shape
            )
            outputs, logabsdet = transform(inputs)
            with self.subTest(shape=shape):
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_inverse(self):
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.AdditiveCouplingTransform, shape
            )
            outputs, logabsdet = transform(inputs)
            with self.subTest(shape=shape):
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_inverse_are_consistent(self):
        self.eps = 1e-6
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.AdditiveCouplingTransform, shape
            )
            with self.subTest(shape=shape):
                self.assert_forward_inverse_are_consistent(transform, inputs)


class UMNNTransformTest(TransformTest):
    shapes = [[20], [2, 4, 4]]

    def test_forward(self):
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.UMNNCouplingTransform, shape, integrand_net_layers=[50, 50, 50],
                cond_size=20,
                nb_steps=20,
                solver="CC"
            )
            outputs, logabsdet = transform(inputs)
            with self.subTest(shape=shape):
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...])

    def test_inverse(self):
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.UMNNCouplingTransform, shape, integrand_net_layers=[50, 50, 50],
                cond_size=20,
                nb_steps=20,
                solver="CC"
            )
            outputs, logabsdet = transform(inputs)
            with self.subTest(shape=shape):
                self.assert_tensor_is_good(outputs, [batch_size] + shape)
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...])

    def test_forward_inverse_are_consistent(self):
        self.eps = 1e-6
        for shape in self.shapes:
            inputs = torch.randn(batch_size, *shape)
            transform, mask = create_coupling_transform(
                coupling.UMNNCouplingTransform, shape, integrand_net_layers=[50, 50, 50],
                cond_size=20,
                nb_steps=20,
                solver="CC"
            )
            with self.subTest(shape=shape):
                self.assert_forward_inverse_are_consistent(transform, inputs)


class PiecewiseCouplingTransformTest(TransformTest):
    classes = [
        coupling.PiecewiseLinearCouplingTransform,
        coupling.PiecewiseQuadraticCouplingTransform,
        coupling.PiecewiseCubicCouplingTransform,
        coupling.PiecewiseRationalQuadraticCouplingTransform,
    ]

    shapes = [[20], [2, 4, 4]]

    def test_forward(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape)
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_forward_unconstrained(self):
        batch_size = 10
        for shape in self.shapes:
            for cls in self.classes:
                inputs = 3.0 * torch.randn(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape, tails="linear")
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_inverse(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape)
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_inverse_unconstrained(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = 3.0 * torch.randn(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape, tails="linear")
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_forward_inverse_are_consistent(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape)
                with self.subTest(cls=cls, shape=shape):
                    self.eps = 1e-4  # TODO: can do better?
                    self.assert_forward_inverse_are_consistent(transform, inputs)

    def test_forward_inverse_are_consistent_unconstrained(self):
        self.eps = 1e-5
        for shape in self.shapes:
            for cls in self.classes:
                inputs = 3.0 * torch.randn(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape, tails="linear")
                with self.subTest(cls=cls, shape=shape):
                    self.eps = 1e-4  # TODO: can do better?
                    self.assert_forward_inverse_are_consistent(transform, inputs)

    def test_forward_unconditional(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                img_shape = shape[1:] if len(shape) > 1 else None
                transform, mask = create_coupling_transform(
                    cls, shape, apply_unconditional_transform=True, img_shape=img_shape
                )
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertNotEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )


if __name__ == "__main__":
    unittest.main()

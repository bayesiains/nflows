"""Tests for the PyTorch utility functions."""

import unittest

import torch

from nflows.utils import torchutils
import torchtestcase


class TorchUtilsTest(torchtestcase.TorchTestCase):
    def test_split_leading_dim(self):
        x = torch.randn(24, 5)
        self.assertEqual(torchutils.split_leading_dim(x, [-1]), x)
        self.assertEqual(torchutils.split_leading_dim(x, [2, -1]), x.view(2, 12, 5))
        self.assertEqual(
            torchutils.split_leading_dim(x, [2, 3, -1]), x.view(2, 3, 4, 5)
        )
        with self.assertRaises(Exception):
            self.assertEqual(torchutils.split_leading_dim(x, []), x)
        with self.assertRaises(Exception):
            self.assertEqual(torchutils.split_leading_dim(x, [5, 5]), x)

    def test_merge_leading_dims(self):
        x = torch.randn(2, 3, 4, 5)
        self.assertEqual(torchutils.merge_leading_dims(x, 1), x)
        self.assertEqual(torchutils.merge_leading_dims(x, 2), x.view(6, 4, 5))
        self.assertEqual(torchutils.merge_leading_dims(x, 3), x.view(24, 5))
        self.assertEqual(torchutils.merge_leading_dims(x, 4), x.view(120))
        with self.assertRaises(Exception):
            torchutils.merge_leading_dims(x, 0)
        with self.assertRaises(Exception):
            torchutils.merge_leading_dims(x, 5)

    def test_split_merge_leading_dims_are_consistent(self):
        x = torch.randn(2, 3, 4, 5)
        y = torchutils.split_leading_dim(torchutils.merge_leading_dims(x, 1), [2])
        self.assertEqual(y, x)
        y = torchutils.split_leading_dim(torchutils.merge_leading_dims(x, 2), [2, 3])
        self.assertEqual(y, x)
        y = torchutils.split_leading_dim(torchutils.merge_leading_dims(x, 3), [2, 3, 4])
        self.assertEqual(y, x)
        y = torchutils.split_leading_dim(
            torchutils.merge_leading_dims(x, 4), [2, 3, 4, 5]
        )
        self.assertEqual(y, x)

    def test_repeat_rows(self):
        x = torch.randn(2, 3, 4, 5)
        self.assertEqual(torchutils.repeat_rows(x, 1), x)
        y = torchutils.repeat_rows(x, 2)
        self.assertEqual(y.shape, torch.Size([4, 3, 4, 5]))
        self.assertEqual(x[0], y[0])
        self.assertEqual(x[0], y[1])
        self.assertEqual(x[1], y[2])
        self.assertEqual(x[1], y[3])
        with self.assertRaises(Exception):
            torchutils.repeat_rows(x, 0)

    def test_logabsdet(self):
        size = 10
        matrix = torch.randn(size, size)
        logabsdet = torchutils.logabsdet(matrix)
        logabsdet_ref = torch.log(torch.abs(matrix.det()))
        self.eps = 1e-6
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_random_orthogonal(self):
        size = 100
        matrix = torchutils.random_orthogonal(size)
        self.assertIsInstance(matrix, torch.Tensor)
        self.assertEqual(matrix.shape, torch.Size([size, size]))
        self.eps = 1e-5
        unit = torch.eye(size, size)
        self.assertEqual(matrix @ matrix.t(), unit)
        self.assertEqual(matrix.t() @ matrix, unit)
        self.assertEqual(matrix.t(), matrix.inverse())
        self.assertEqual(torch.abs(matrix.det()), torch.tensor(1.0))

    def test_searchsorted(self):
        bin_locations = torch.linspace(0, 1, 10)  # 9 bins == 10 locations

        left_boundaries = bin_locations[:-1]
        right_boundaries = bin_locations[:-1] + 0.1
        mid_points = bin_locations[:-1] + 0.05

        for inputs in [left_boundaries, right_boundaries, mid_points]:
            with self.subTest(inputs=inputs):
                idx = torchutils.searchsorted(bin_locations[None, :], inputs)
                self.assertEqual(idx, torch.arange(0, 9))

    def test_searchsorted_arbitrary_shape(self):
        shape = [2, 3, 4]
        bin_locations = torch.linspace(0, 1, 10).repeat(*shape, 1)
        inputs = torch.rand(*shape)
        idx = torchutils.searchsorted(bin_locations, inputs)
        self.assertEqual(idx.shape, inputs.shape)


if __name__ == "__main__":
    unittest.main()

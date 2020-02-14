"""Tests for permutations."""

import unittest

import torch

from nflows.transforms import permutations
from tests.transforms.transform_test import TransformTest


class PermutationTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        features = 100
        inputs = torch.randn(batch_size, features)
        permutation = torch.randperm(features)
        transform = permutations.Permutation(permutation)
        outputs, logabsdet = transform(inputs)
        self.assert_tensor_is_good(outputs, [batch_size, features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, inputs[:, permutation])
        self.assertEqual(logabsdet, torch.zeros([batch_size]))

    def test_inverse(self):
        batch_size = 10
        features = 100
        inputs = torch.randn(batch_size, features)
        permutation = torch.randperm(features)
        transform = permutations.Permutation(permutation)
        temp, _ = transform(inputs)
        outputs, logabsdet = transform.inverse(temp)
        self.assert_tensor_is_good(outputs, [batch_size, features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, inputs)
        self.assertEqual(logabsdet, torch.zeros([batch_size]))

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 100
        inputs = torch.randn(batch_size, features)
        transforms = [
            permutations.Permutation(torch.randperm(features)),
            permutations.RandomPermutation(features),
            permutations.ReversePermutation(features),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()

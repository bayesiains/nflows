import unittest

import torch

from nflows.transforms.reshape import SqueezeTransform
from tests.transforms.transform_test import TransformTest


class SqueezeTransformTest(TransformTest):
    def setUp(self):
        self.transform = SqueezeTransform()

    def test_forward(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            with self.subTest(shape=shape):
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                outputs, logabsdet = self.transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, c * 4, h // 2, w // 2])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_values(self):
        inputs = torch.arange(1, 17, 1).long().view(1, 1, 4, 4)
        outputs, _ = self.transform(inputs)

        def assert_channel_equal(channel, values):
            self.assertEqual(outputs[0, channel, ...], torch.LongTensor(values))

        assert_channel_equal(0, [[1, 3], [9, 11]])
        assert_channel_equal(1, [[2, 4], [10, 12]])
        assert_channel_equal(2, [[5, 7], [13, 15]])
        assert_channel_equal(3, [[6, 8], [14, 16]])

    def test_forward_wrong_shape(self):
        batch_size = 10
        for shape in [[32, 3, 3], [32, 5, 5], [32, 4]]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.transform(inputs)

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            with self.subTest(shape=shape):
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                self.assert_forward_inverse_are_consistent(self.transform, inputs)

    def test_inverse_wrong_shape(self):
        batch_size = 10
        for shape in [[3, 4, 4], [33, 4, 4], [32, 4]]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.transform.inverse(inputs)


if __name__ == "__main__":
    unittest.main()

"""Tests for the orthogonal transforms."""

import unittest

import torch
import numpy as np
import math
from nflows.transforms import orthogonal
from nflows.utils import torchutils
from tests.transforms.transform_test import TransformTest


class HouseholderSequenceTest(TransformTest):
    def test_forward(self):
        features = 100
        batch_size = 50

        for num_transforms in [1, 2, 11, 12]:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                inputs = torch.randn(batch_size, features)
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.eps = 1e-5
                self.assertEqual(outputs, inputs @ matrix.t())
                self.assertEqual(
                    logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size)
                )

    def test_inverse(self):
        features = 100
        batch_size = 50

        for num_transforms in [1, 2, 11, 12]:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                inputs = torch.randn(batch_size, features)
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.eps = 1e-5
                self.assertEqual(outputs, inputs @ matrix)
                self.assertEqual(
                    logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size)
                )

    def test_matrix(self):
        features = 100

        for num_transforms in [1, 2, 11, 12]:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                self.assert_tensor_is_good(matrix, [features, features])
                self.eps = 1e-5
                self.assertEqual(matrix @ matrix.t(), torch.eye(features, features))
                self.assertEqual(matrix.t() @ matrix, torch.eye(features, features))
                self.assertEqual(matrix.t(), torch.inverse(matrix))
                det_ref = torch.tensor(1.0 if num_transforms % 2 == 0 else -1.0)
                self.assertEqual(matrix.det(), det_ref)

    def test_forward_inverse_are_consistent(self):
        features = 100
        batch_size = 50
        inputs = torch.randn(batch_size, features)
        transforms = [
            orthogonal.HouseholderSequence(
                features=features, num_transforms=num_transforms
            )
            for num_transforms in [1, 2, 11, 12]
        ]
        self.eps = 1e-5
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


    def test_forward_full_orthogonal(self):
        features = 100
        batch_size = 50
        transform = orthogonal.FullOrthogonalTransform(
            features=features
        )
        matrix = transform.matrix()
        inputs = torch.randn(batch_size, features)
        outputs, logabsdet = transform(inputs)
        self.assert_tensor_is_good(outputs, [batch_size, features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.eps = 1e-5
        self.assertEqual(outputs, inputs @ matrix.t())
        self.assertEqual(outputs, inputs @ transform.W)
        self.assertEqual(
            logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size)
        )
        self.assert_forward_inverse_are_consistent(transform, inputs)

    def test_inverse_full_orthogonal(self):
        features = 100
        batch_size = 50
        transform = orthogonal.FullOrthogonalTransform(
            features=features
        )
        matrix = transform.matrix()
        inputs = torch.randn(batch_size, features)
        outputs, logabsdet = transform.inverse(inputs)
        self.assert_tensor_is_good(outputs, [batch_size, features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.eps = 1e-5
        self.assertEqual(outputs, inputs @ matrix)
        self.assertEqual(
            logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size)
        )
        self.assert_forward_inverse_are_consistent(transform, inputs)


    def verify_qr_of_orthogonal_matrix(self, U, n):
        Q, R, U2 = orthogonal.householder_qr(U)
        diffr = (R - torch.eye(n)).abs().max().item()
        # When doing *this* QR decomposition of an Orthogonal Matrix, we expect resulting R to be the identity matrix
        # as in the proof
        self.assertTrue(diffr < 2e-5)
        diffi = (R - torch.eye(n)).abs().max().item()
        self.assertTrue(diffi < 1e-5)
        diffq = (Q - U).abs().max().item()
        self.assertTrue(diffq < 1e-5)
        # Now make sure that the Matrix Q is actually the product of our householder matrices
        Q2 = torch.eye(n)
        for i in range(U2.shape[0]):
            Q2 = torch.mm(Q2, orthogonal.create_hh_matrix(U2[i, :]))
        diffq2 = (Q - Q2).abs().max().item()
        self.assertTrue(diffq2 < 1e-5)
        # Now make sure that the Matrix Q can be constructed using the simplified householder transformation
        # Using vector products (note that the order needs to be inversed)
        Q3 = torch.eye(n)
        for i in reversed(range(U2.shape[0])):
            Q3 = orthogonal.simplified_hh_vector_product(U2[i, :], Q3)
        diffq3 = (Q - Q3).abs().max().item()
        self.assertTrue(diffq3 < 1e-5)

    def test_householder_qr(self):
        # Test our implementation of the QR Decomposition of Orthogonal Matrices
        # into Householder Reflection Vecors

        # First, to check some boundary conditions, we try some trivial orthogonal matrices
        self.verify_qr_of_orthogonal_matrix(torch.eye(1), 1)
        self.verify_qr_of_orthogonal_matrix(torch.eye(2), 2)
        self.verify_qr_of_orthogonal_matrix(torch.eye(3), 3)
        self.verify_qr_of_orthogonal_matrix(torch.FloatTensor([[0, 1], [1, 0]]), 2)
        self.verify_qr_of_orthogonal_matrix(torch.FloatTensor([[3, 4], [-4, 3]]) * 0.2, 2)

        # Now we try a lot of rotation matrices
        for alpha in np.linspace(0.0, 2.0, 36):
            a = alpha * math.pi
            rmatrix = torch.FloatTensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            self.verify_qr_of_orthogonal_matrix(rmatrix, 2)

        # And now a lot of random 20 dimensional matrices
        for k in range(10):
            n = 20
            A = torch.randn((n, n))
            U, S, V = torch.svd(A)
            # Now U and V are random orthogonal Matrices
            self.verify_qr_of_orthogonal_matrix(U, n)
            self.verify_qr_of_orthogonal_matrix(V, n)


    def test_orthogonal_transform_W_setter(self):
        a = torch.randn((4, 4))
        u, s, v = torch.svd(a)
        ot = orthogonal.FullOrthogonalTransform(4, bias=False)
        ot.W = u  # This uses a setter method

        # Check that we have actually set W, and that the reconstruction into property W works
        wr = ot.W
        self.assertTrue((u - wr).abs().max().item() < 1e-5)

        # Check that forward really applies the orthogonal transformation we specified
        wr2, logabsdet = ot.forward(torch.eye(4))
        self.assertTrue((u - wr2).abs().max().item() < 1e-5)

        # Check that forward really applies the orthogonal transformation we specified
        wr3_t, logabsdet = ot.inverse(torch.eye(4))
        wr3 = wr3_t.t()
        self.assertTrue((u - wr3).abs().max().item() < 1e-5)

    def test_orthogonal_transform(self):
        n = 20
        for bias in [True, False]:
            for i in range(10):
                a = torch.randn((10, n))
                ot = orthogonal.FullOrthogonalTransform(n, bias=bias)
                b, logabsdet = ot.forward(a)
                self.assertFalse(np.all(np.isclose(a.detach().numpy(), b.detach().numpy(), atol=1e-5)))
                ar, logabsdet = ot.inverse(b)
                self.assertTrue(np.all(np.isclose(a.detach().numpy(), ar.detach().numpy(), atol=1e-5)))
                self.assertAlmostEqual(torch.slogdet(ot.W)[1].item(), 0.0,
                                       delta=1e-5)  # Determinant of Orthogonal Matrix needs to be close to 1 or -1
                W = ot.W
                Wt = ot.W.t()
                Winv = torch.inverse(W)
                self.assertTrue(np.all(np.isclose(Wt.detach().numpy(), Winv.detach().numpy(), atol=1e-5)))
            for i in range(10):
                a = torch.randn((10, n))
                ot = orthogonal.FullOrthogonalTransform(n, bias=bias)
                b, logabsdet = ot.inverse(a)
                self.assertFalse(np.all(np.isclose(a.detach().numpy(), b.detach().numpy(), atol=1e-5)))
                ar, logabsdet = ot.forward(b)
                self.assertTrue(np.all(np.isclose(a.detach().numpy(), ar.detach().numpy(), atol=1e-5)))
                self.assertAlmostEqual(torch.slogdet(ot.W)[1].item(), 0.0,
                                       delta=1e-5)  # Determinant of Orthogonal Matrix needs to be close to 1 or -1
                W = ot.W
                Wt = ot.W.t()
                Winv = torch.inverse(W)
                self.assertTrue(np.all(np.isclose(Wt.detach().numpy(), Winv.detach().numpy(), atol=1e-6)))

    def test_orthogonal_transform2d(self):
        for bias in [True, False]:
            for i in range(10):
                a = torch.randn((10, 5, 10, 10))
                ot = orthogonal.FullOrthogonalTransform2D(5, bias=bias)
                b, logabsdet = ot.forward(a)
                self.assertFalse(np.all(np.isclose(a.detach().numpy(), b.detach().numpy(), atol=1e-6)))
                ar, logabsdet = ot.inverse(b)
                self.assertTrue(np.all(np.isclose(a.detach().numpy(), ar.detach().numpy(), atol=1e-5)))
                self.assertAlmostEqual(torch.slogdet(ot.W)[1].item(), 0.0,
                                       delta=2e-6)  # Determinant of Orthogonal Matrix needs to be close to 1 or -1
                W = ot.W
                Wt = ot.W.t()
                Winv = torch.inverse(W)
                self.assertTrue(np.all(np.isclose(Wt.detach().numpy(), Winv.detach().numpy(), atol=1e-5)))
            for i in range(10):
                a = torch.randn((10, 5, 10, 10))
                ot = orthogonal.FullOrthogonalTransform2D(5, bias=bias)
                b, logabsdet = ot.inverse(a)
                self.assertFalse(np.all(np.isclose(a.detach().numpy(), b.detach().numpy(), atol=1e-5)))
                ar, logabsdet = ot.forward(b)
                self.assertTrue(np.all(np.isclose(a.detach().numpy(), ar.detach().numpy(), atol=1e-5)))
                self.assertAlmostEqual(torch.slogdet(ot.W)[1].item(), 0.0,
                                       delta=2e-6)  # Determinant of Orthogonal Matrix needs to be close to 1 or -1
                W = ot.W
                Wt = ot.W.t()
                Winv= torch.inverse(W)
                self.assertTrue(np.all(np.isclose(Wt.detach().numpy(), Winv.detach().numpy(), atol=3e-6)))

if __name__ == "__main__":
    unittest.main()

import torch
import torchtestcase

from nflows.transforms import splines


class LinearSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_pdf = torch.randn(*shape, num_bins)

        def call_spline_fn(inputs, inverse=False):
            return splines.linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse
            )

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))


class UnconstrainedLinearSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_pdf = torch.randn(*shape, num_bins)

        def call_spline_fn(inputs, inverse=False):
            return splines.unconstrained_linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse
            )

        inputs = 3 * torch.randn(*shape)  # Note inputs are outside [0,1].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_forward_inverse_are_consistent_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_pdf = torch.randn(*shape, num_bins)

        def call_spline_fn(inputs, inverse=False):
            return splines.unconstrained_linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse, tail_bound=tail_bound
            )

        inputs = torch.sign(torch.randn(*shape)) * (tail_bound + torch.rand(*shape))  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

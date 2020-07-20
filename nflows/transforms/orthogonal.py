"""Implementations of orthogonal transforms."""

import torch
from torch import nn
import torch.nn.functional as F
from nflows.transforms.base import Transform
import nflows.utils.typechecks as check


class HouseholderSequence(Transform):
    """A sequence of Householder transforms.

    This class can be used as a way of parameterizing an orthogonal matrix.
    """

    def __init__(self, features, num_transforms):
        """Constructor.

        Args:
            features: int, dimensionality of the input.
            num_transforms: int, number of Householder transforms to use.

        Raises:
            TypeError: if arguments are not the right type.
        """
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        if not check.is_positive_int(num_transforms):
            raise TypeError("Number of transforms must be a positive integer.")

        super().__init__()
        self.features = features
        self.num_transforms = num_transforms
        # TODO: are randn good initial values?
        # these vectors are orthogonal to the hyperplanes through which we reflect
        # self.q_vectors = nets.Parameter(torch.randn(num_transforms, features))
        # self.q_vectors = nets.Parameter(torch.eye(num_transforms // 2, features))
        import numpy as np

        def tile(a, dim, n_tile):
            if a.nelement() == 0:
                return a
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))

            order_index = torch.Tensor(
                np.concatenate(
                    [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
                )
            ).long()
            return torch.index_select(a, dim, order_index)

        qv = tile(torch.eye(num_transforms // 2, features), 0, 2)
        if np.mod(num_transforms, 2) != 0:  # odd number of transforms, including 1
            qv = torch.cat((qv, torch.zeros(1, features)))
            qv[-1, num_transforms // 2] = 1
        self.q_vectors = nn.Parameter(qv)

    @staticmethod
    def _apply_transforms(inputs, q_vectors):
        """Apply the sequence of transforms parameterized by given q_vectors to inputs.

        Costs O(KDN), where:
        - K is number of transforms
        - D is dimensionality of inputs
        - N is number of inputs

        Args:
            inputs: Tensor of shape [N, D]
            q_vectors: Tensor of shape [K, D]

        Returns:
            A tuple of:
            - A Tensor of shape [N, D], the outputs.
            - A Tensor of shape [N], the log absolute determinants of the total transform.
        """
        squared_norms = torch.sum(q_vectors ** 2, dim=-1)
        outputs = inputs
        for q_vector, squared_norm in zip(q_vectors, squared_norms):
            temp = outputs @ q_vector  # Inner product.
            temp = torch.ger(temp, (2.0 / squared_norm) * q_vector)  # Outer product.
            outputs = outputs - temp
        batch_size = inputs.shape[0]
        logabsdet = torch.zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._apply_transforms(inputs, self.q_vectors)

    def inverse(self, inputs, context=None):
        # Each householder transform is its own inverse, so the total inverse is given by
        # simply performing each transform in the reverse order.
        reverse_idx = torch.arange(self.num_transforms - 1, -1, -1)
        return self._apply_transforms(inputs, self.q_vectors[reverse_idx])

    def matrix(self):
        """Returns the orthogonal matrix that is equivalent to the total transform.

        Costs O(KD^2), where:
        - K is number of transforms
        - D is dimensionality of inputs

        Returns:
            A Tensor of shape [D, D].
        """
        identity = torch.eye(self.features, self.features)
        outputs, _ = self.inverse(identity)
        return outputs


def simplified_hh_vector_product(u, a):
    '''
    Apply Householder Transformation to Vector or Matrix, given reflection Vector
    Args:
        u (torch.Tensor): Householder Reflection Vector of shape (n)
        a (torch.Tensor): Vector of Shape (n) or Matrix of shape (n, *)
    Returns:
        Transformed Vector or Matrix of same shape as a
    '''
    if len(a.shape) == 1:
        a = a.unsqueeze(-1)
    n2 = torch.dot(u, u).sqrt()
    if (n2.item() == 0.0):
        return a
    un = u / n2
    return a - (2.0 * un.unsqueeze(0).mm(a)) * un.unsqueeze(-1)


def create_hh_matrix(v):
    '''
    Create Householder Matrix from Reflection Vector
    :param v: Reflection Vector of shape (n)
    :return: Householder Matrix of shape (n,n)
    '''
    H = torch.eye(v.shape[0])
    n2sq = torch.dot(v, v)
    if n2sq == 0.0:
        return H
    H -= (2 / n2sq) * torch.mm(v.unsqueeze(-1), v.unsqueeze(0))
    return H


def householder_qr(A):
    '''
    Perform QR decomposition based on a constructive inverse of Proof A.1 of Paper
    Zhang: Stabilizing Gradients for Deep Neural Networks via Efficient SVD Parameterization
    see https://arxiv.org/pdf/1803.09327.pdf

    This algorithm ensures that the resulting R has a nonnegative diagonal,
    and we get the proper reflection vectors.

    Args:
        A (torch.Tensor): Matrix to factorize
    Returns:
        Q (torch.Tensor): Orthogonal Matrix Q
        R (torch.Tensor): Upper triangular matrix with positive diagonal
        U (torch.Tensor): Upper Triangular Matrix of Reflection Vectors
    '''

    def _make_householder(a):
        v = a.clone()  # - np.linalg.norm_type(a)(a[0] + np.copysign(np.linalg.norm_type(a), a[0]))
        v[0] -= a.norm(2)
        return v

    m, n = A.shape
    Q = torch.eye(m)
    U = torch.eye(m)
    for i in range(n - (m == n)):
        U[i, i:] = _make_householder(A[i:, i])
        H = create_hh_matrix(U[i, :])
        Q = torch.mm(Q, H)
        A = simplified_hh_vector_product(U[i, :], A)

    # Exception for n=1 as in proof
    # except that we apply it to the last index, instead of the first.
    if m == n:
        i = n - 1
        if A[i, i] > 0.0:
            U[i, i] = 0.0
        else:
            U[i, i] = 1.0
        H = create_hh_matrix(U[i, :])
        Q = torch.mm(Q, H)
        R = torch.mm(H, A)
    else:
        R = A
    return Q, R, U


class FullOrthogonalTransform(Transform):
    '''
    This module implements an efficiently parameterized & quickly invertible Orthogonal Linear Transformation
    based on a product of Householder Matrices (or Reflectors)

    The transform is equivalent to multiplication with an orthogonal matrix
    (accessible as commputed property W of this module ) and optional subsequent addition of a bias term.

    The image of the product of these reflectors is the entire manifold of orthogonal matrices,
    so that efficient backprop is possible while maintaining the Orthogonality Property of the Transform.

    It is possible to assign an orthogonal matrix to the W property, which will then be decomposed into
    the corresponding householder reflections via qr decomposition.
    '''

    weight: nn.Parameter
    bias: nn.Parameter
    n: int
    m: int
    h1: float

    def __init__(self, features: int, bias=False):
        """
        Orthogonal Linear Transformation, efficiently parameterized & invertible.

        :param features: Dimensionality of Transform ( equivalent orthogonal Matrix W will have dimensions features x features )
        :param bias: Whether to add / subtract a bias term. Defaults to false
        """
        super().__init__()
        self.h1 = 1.0  # Constant for lowest right coefficient of last householder reflection matrix, has to be .0 or 1.0. Only used if features==num_transforms
        self.n = features
        self.m = features # number of householder reflections. Fixed at number of features in this class.
        # Full parameter matrix, which needs to be constrained further to a lower triangular form (see property U below)
        self.weight = nn.Parameter(torch.randn((features, self.m)) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.randn((1, features)) * 0.01)
        else:
            self.register_parameter('bias', None)
        self.register_buffer('_eye_n', torch.eye(self.n, requires_grad=False))

    @property
    def U(self):
        '''
        Constrained parameter Matrix U.

        This is weight constrained to lower triangular, with U[-1,-1] set to h1 if n==m
        see Mhammedi: Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections
        '''
        U = self.weight.triu()
        if self.n == self.m:
            U.data[-1, -1] = self.h1
        return U

    @property
    def W(self):
        '''
        Orthogonal Matrix W, computed on-the-fly, representing the forward transformation without bias term.

        Returns:
            A Tensor of shape [D, D].
        '''
        return self._forward(self._eye_n)

    def matrix(self):
        """Returns the orthogonal matrix that is equivalent to the inverse transform.

        Returns:
            A Tensor of shape [D, D].
        """
        return self._inverse(self._eye_n)

    @W.setter
    def W(self, new_W):
        """
        Assign a valid orthogonal matrix to this transformation. This matrix will be decomposed into a
        sequence of Householder reflections.

        This call will fail with an assertion error if the assigned matrix is not orthogonal,
        or if n!=m in the constructor
        :param new_W: Torch
        """
        # Check preconditions
        with torch.no_grad():
            assert (self.n == self.m)
            assert (len(new_W.shape) == 2)
            assert (self.n == new_W.shape[0])
            assert (self.n == new_W.shape[1])
            assert (torch.slogdet(new_W)[1].abs().item() < 1e-5)  # Simple check for orthogonality
            Q, R, U = householder_qr(new_W.t())
            self.weight.data.copy_(U)
            self.h1 = U[-1, -1].item()

    def _forward(self, x):
        h = x.t().contiguous()
        U = self.U
        for i in reversed(range(self.m)):
            h = simplified_hh_vector_product(U[i, :], h)

        return h.t().contiguous()

    def _inverse(self, x):
        '''
        Peforms inverted Orthogonal Transform of x
        equivalent to self.W.t().matmul(x), but slightly more efficient

        Args:
            x ( torch.Tensor): Input tensor to transform
        Returns:
            Transformed tensor of same shape as x
        '''
        h = x.t().contiguous()
        U = self.U
        for i in range(self.m):
            h = simplified_hh_vector_product(U[i, :], h)
        return h.t().contiguous()

    def forward(self, x, context=None):
        logabsdet = torch.zeros(x.shape[0])
        return self._forward(x), logabsdet

    def inverse(self, x, context=None):
        inverse = self._inverse(x)
        logabsdet = torch.zeros(x.shape[0])
        return inverse, logabsdet

class FullOrthogonalTransform2D(FullOrthogonalTransform):
    """
    Full 1x1 convolution operation with an orthogonal matrix backed by a sequence of householder reflectors.
    See class FullOrthogonalTransform for details.

    Allows to manually set and retrieve weight matrix "W" and optional bias vector "bias"
    """

    def _forward_conv(self, x):
        '''
        Performs forward Orthogonal Transform of x as a pixelwise 2D convolution

        Args:
            x (torch.Tensor): Tensor to transform
        Returns:
            Transformed tensor of same shape as x
        '''
        W = self.W.view(self.n, self.n, 1, 1)
        if self.bias is not None:
            bias = self.bias.squeeze()
        else:
            bias = None
        return F.conv2d(x, W, bias, (1, 1), (0, 0), (1, 1), 1)

    def _inverse_conv(self, x):
        '''
        Peforms inverted Orthogonal Transform of x
        equivalent to self.W.t().matmul(x), but slightly more efficient
        Args:
            x (torch.Tensor): Input Tensor to be transformed
        Returns:
            Transformed tensor of same shape as x
        '''
        Wt = self.W.t().view(self.n, self.n, 1, 1)
        if self.bias is not None:
            bias = self.bias.view(1, self.bias.shape[1], 1, 1).contiguous()
            return F.conv2d(x - bias, Wt, None, (1, 1), (0, 0), (1, 1), 1)
        else:
            return F.conv2d(x, Wt, None, (1, 1), (0, 0), (1, 1), 1)

    def forward(self, x, context=None):
        logabsdet = torch.zeros(x.shape[0])
        return self._forward_conv(x), logabsdet

    def inverse(self, x, context=None):
        logabsdet = torch.zeros(x.shape[0])
        return self._inverse_conv(x), logabsdet


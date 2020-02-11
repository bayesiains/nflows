import numpy as np


class MCMCSampler:
    """
    Superclass for MCMC samplers.
    """

    def __init__(self, x, lp_f, thin):
        """
        :param x: initial state
        :param lp_f: function that returns the log prob
        :param thin: amount of thinning; if None, no thinning
        """

        self.x = np.array(x, dtype=float)
        self.lp_f = lp_f
        self.L = lp_f(self.x)
        self.thin = 1 if thin is None else thin
        self.n_dims = self.x.size if self.x.ndim == 1 else self.x.shape[1]

    def set_state(self, x):
        """
        Sets the state of the chain to x.
        """

        self.x = np.array(x, dtype=float)
        self.L = self.lp_f(self.x)

    def gen(self, n_samples):
        """
        Generates MCMC samples. Should be implemented in a subclass.
        """
        raise NotImplementedError("Should be implemented as a subclass.")

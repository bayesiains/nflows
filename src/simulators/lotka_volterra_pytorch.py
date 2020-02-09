import numpy as np
import os
import pickle
import torch

import utils

from summarizers import LotkaVolterraSummarizer
from .simulator import Simulator
from .markov_jump_process import MarkovJumpProcess, SimTooLongException

observation_dim = 9
parameter_dim = 4


class LotkaVolterraMarkovJumpProcess(MarkovJumpProcess):
    """
    The Lotka-Volterra implementation of the Markov Jump Process.
    """

    def _compute_propensities(self):

        x, y = self._state  # predator and prey populations
        xy = x * y
        return np.array([xy, x, y, xy])

    def _do_reaction(self, reaction):

        # Predator is born.
        if reaction == 0:
            self._state[0] += 1

        # Predator dies.
        elif reaction == 1:
            self._state[0] -= 1

        # Prey is born.
        elif reaction == 2:
            self._state[1] += 1

        # Prey is eaten by predator.
        elif reaction == 3:
            self._state[1] -= 1

        else:
            raise ValueError("Unknown reaction.")


class LotkaVolterraSimulator(Simulator):
    """
    Implementation of Lotka-Volterra stochastic predator-prey dynamics using a Markov Jump
    Process simulated using the Gillespie algorithm.
    Code follows setup in
    'Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
    Density Estimation'
    Papmakarios & Murray
    NeurIPS 2016
    https://arxiv.org/abs/1605.06376
    """

    def __init__(self, summarize_observations=True):

        super().__init__()

        self._initial_populations = [50, 100]
        self._dt = 0.2
        self._duration = 30
        self._max_num_steps = 10000
        self._jump_process = LotkaVolterraMarkovJumpProcess(
            initial_state=self._initial_populations, parameters=None
        )
        self._summarize_observations = summarize_observations
        if summarize_observations:
            self._summarizer = LotkaVolterraSummarizer()
        self._has_been_used = False

    def simulate(self, parameters):

        parameters = utils.tensor2numpy(parameters)
        parameters = np.exp(parameters)

        observations = []

        for i, parameter in enumerate(parameters):
            try:
                self._jump_process.reset(self._initial_populations, parameter)
                states = self._jump_process.simulate_for_time(
                    self._dt, self._duration, max_n_steps=self._max_num_steps
                )
                observations.append(torch.Tensor(states.flatten()))
            except SimTooLongException:
                observations.append(None)
            self.num_total_simulations += 1

        if self._summarize_observations:
            return self._summarizer(observations)

        return observations

    def get_ground_truth_parameters(self):
        """
        Ground truth parameters as given in
        'Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
        Density Estimation'

        :return:
        """
        return torch.log(torch.Tensor([0.01, 0.5, 1.0, 0.01]))

    def get_ground_truth_observation(self):
        path = os.path.join(utils.get_data_root(), "lotka-volterra", "obs_stats.pkl")
        with open(path, "rb") as file:
            true_observation = pickle.load(file, encoding="bytes")
        return torch.Tensor(true_observation)

    @property
    def parameter_dim(self):
        return 4

    @property
    def observation_dim(self):
        if self._summarize_observations:
            return 9
        else:
            return 32

    @property
    def parameter_plotting_limits(self):
        return [-5, 2]

    @property
    def name(self):
        return "lotka-volterra"

    def _get_prior_parameters_observations(self):
        self._has_been_used = True

        parameters = np.load(
            os.path.join(
                utils.get_data_root(), "lotka-volterra", "prior-parameters.npy"
            )
        )

        observations = np.load(
            os.path.join(
                utils.get_data_root(), "lotka-volterra", "prior-observations.npy"
            )
        )

        ix = np.random.permutation(range(parameters.shape[0]))

        return parameters[ix], observations[ix]

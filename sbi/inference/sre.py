import numpy as np
import os
import torch

import sbi.simulators as simulators
import sbi.utils as utils

from copy import deepcopy
from matplotlib import pyplot as plt
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import distributions, multiprocessing as mp, nn, optim
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sbi.mcmc import Slice, SliceSampler


if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")


class SRE:
    """
    Implementation 'Sequential Ratio Estimation', as presented in
    'Likelihood-free MCMC with Amortized Approximate Likelihood Ratios'
    Hermans et al.
    Pre-print 2019
    https://arxiv.org/abs/1903.04057
    """

    def __init__(
        self,
        simulator,
        prior,
        true_observation,
        classifier,
        num_atoms=-1,
        mcmc_method="slice-np",
        summary_net=None,
        retrain_from_scratch_each_round=False,
        summary_writer=None,
    ):
        """
        :param simulator: Python object with 'simulate' method which takes a torch.Tensor
        of parameter values, and returns a simulation result for each parameter as a torch.Tensor.
        :param prior: Distribution object with 'log_prob' and 'sample' methods.
        :param true_observation: torch.Tensor containing the observation x0 for which to
        perform inference on the posterior p(theta | x0).
        :param classifier: Binary classifier in the form of an nets.Module.
        Takes as input (x, theta) pairs and outputs pre-sigmoid activations.
        :param num_atoms: int
            Number of atoms to use for classification.
            If -1, use all other parameters in minibatch.
        :param summary_net: Optional network which may be used to produce feature vectors
        f(x) for high-dimensional observations.
        :param retrain_from_scratch_each_round: Whether to retrain the conditional density
        estimator for the posterior from scratch each round.
        """

        self._simulator = simulator
        self._true_observation = true_observation
        self._classifier = classifier
        self._prior = prior

        assert isinstance(num_atoms, int), "Number of atoms must be an integer."
        self._num_atoms = num_atoms

        self._mcmc_method = mcmc_method

        # We may want to summarize high-dimensional observations.
        # This may be either a fixed or learned transformation.
        if summary_net is None:
            self._summary_net = nn.Identity()
        else:
            self._summary_net = summary_net

        # Defining the potential function as an object means Pyro's MCMC scheme
        # can pickle it to be used across multiple chains in parallel, even if
        # the potential function requires evaluating a neural likelihood as is the
        # case here.
        self._potential_function = NeuralPotentialFunction(
            classifier, prior, true_observation
        )

        # TODO: decide on Slice Sampling implementation
        target_log_prob = (
            lambda parameters: self._classifier(
                torch.cat((torch.Tensor(parameters), self._true_observation)).reshape(
                    1, -1
                )
            ).item()
            + self._prior.log_prob(torch.Tensor(parameters)).sum().item()
        )
        self._classifier.eval()
        self.posterior_sampler = SliceSampler(
            utils.tensor2numpy(self._prior.sample((1,))).reshape(-1),
            lp_f=target_log_prob,
            thin=10,
        )
        self._classifier.train()

        self._retrain_from_scratch_each_round = retrain_from_scratch_each_round
        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        if retrain_from_scratch_each_round:
            self._untrained_classifier = deepcopy(classifier)
        else:
            self._untrained_classifier = None

        # Need somewhere to store (parameter, observation) pairs from each round.
        self._parameter_bank, self._observation_bank = [], []

        # Each SRE run has an associated log directory for TensorBoard output.
        if summary_writer is None:
            log_dir = os.path.join(
                utils.get_log_root(), "sre", simulator.name, utils.get_timestamp()
            )
            self._summary_writer = SummaryWriter(log_dir)
        else:
            self._summary_writer = summary_writer

        # Each run also has a dictionary of summary statistics which are populated
        # over the course of training.
        self._summary = {
            "mmds": [],
            "median-observation-distances": [],
            "negative-log-probs-true-parameters": [],
            "neural-net-fit-times": [],
            "mcmc-times": [],
            "epochs": [],
            "best-validation-log-probs": [],
        }

    def run_inference(self, num_rounds, num_simulations_per_round):
        """
        This runs SRE for num_rounds rounds, using num_simulations_per_round calls to
        the simulator per round.

        :param num_rounds: Number of rounds to run.
        :param num_simulations_per_round: Number of simulator calls per round.
        :return: None
        """

        round_description = ""
        tbar = tqdm(range(num_rounds))
        for round_ in tbar:

            tbar.set_description(round_description)

            # Generate parameters from prior in first round, and from most recent posterior
            # estimate in subsequent rounds.
            if round_ == 0:
                parameters, observations = simulators.simulation_wrapper(
                    simulator=self._simulator,
                    parameter_sample_fn=lambda num_samples: self._prior.sample(
                        (num_samples,)
                    ),
                    num_samples=num_simulations_per_round,
                )
            else:
                parameters, observations = simulators.simulation_wrapper(
                    simulator=self._simulator,
                    parameter_sample_fn=lambda num_samples: self.sample_posterior(
                        num_samples
                    ),
                    num_samples=num_simulations_per_round,
                )

            # Store (parameter, observation) pairs.
            self._parameter_bank.append(torch.Tensor(parameters))
            self._observation_bank.append(torch.Tensor(observations))

            # Fit posterior using newly aggregated data set.
            self._fit_classifier()

            # Update description for progress bar.
            round_description = (
                f"-------------------------\n"
                f"||||| ROUND {round_ + 1} STATS |||||:\n"
                f"-------------------------\n"
                f"Epochs trained: {self._summary['epochs'][-1]}\n"
                f"Best validation performance: {self._summary['best-validation-log-probs'][-1]:.4f}\n\n"
            )

            # Update tensorboard and summary dict.
            self._summarize(round_)

    def sample_posterior(self, num_samples, thin=10):
        """
        Samples from posterior for true observation q(theta | x0) ~ r(x0, theta) p(theta)
        using most recent ratio estimate r(x0, theta) with MCMC.

        :param num_samples: Number of samples to generate.
        :param mcmc_method: Which MCMC method to use ['metropolis-hastings', 'slice', 'hmc', 'nuts']
        :param thin: Generate (num_samples * thin) samples in total, then select every
        'thin' sample.
        :return: torch.Tensor of shape [num_samples, parameter_dim]
        """

        # Always sample in eval mode.
        self._classifier.eval()

        if self._mcmc_method == "slice-np":
            self.posterior_sampler.gen(20)
            samples = torch.Tensor(self.posterior_sampler.gen(num_samples))

        else:
            if self._mcmc_method == "slice":
                kernel = Slice(potential_function=self._potential_function)
            elif self._mcmc_method == "hmc":
                kernel = HMC(potential_fn=self._potential_function)
            elif self._mcmc_method == "nuts":
                kernel = NUTS(potential_fn=self._potential_function)
            else:
                raise ValueError(
                    "'mcmc_method' must be one of ['slice', 'hmc', 'nuts']."
                )
            num_chains = mp.cpu_count() - 1

            initial_params = self._prior.sample((num_chains,))
            sampler = MCMC(
                kernel=kernel,
                num_samples=(thin * num_samples) // num_chains + num_chains,
                warmup_steps=200,
                initial_params={"": initial_params},
                num_chains=num_chains,
                mp_context="spawn",
            )
            sampler.run()
            samples = next(iter(sampler.get_samples().values())).reshape(
                -1, self._simulator.parameter_dim
            )

            samples = samples[::thin][:num_samples]
            assert samples.shape[0] == num_samples

        # Back to training mode.
        self._classifier.train()

        return samples

    def _fit_classifier(
        self,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
    ):
        """
        Trains the classifier by maximizing a Bernoulli likelihood which distinguishes
        between jointly distributed (parameter, observation) pairs and randomly chosen
        (parameter, observation) pairs.
        Uses early stopping on a held-out validation set as a terminating condition.

        :param batch_size: Size of batch to use for training.
        :param learning_rate: Learning rate for Adam optimizer.
        :param validation_fraction: The fraction of data to use for validation.
        :param stop_after_epochs: The number of epochs to wait for improvement on the
        validation set before terminating training.
        :return: None
        """

        # Get total number of training examples.
        num_examples = torch.cat(self._parameter_bank).shape[0]

        # Select random train and validation splits from (parameter, observation) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            torch.cat(self._parameter_bank), torch.cat(self._observation_bank)
        )

        # Create train and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, num_examples - num_training_examples),
            shuffle=False,
            drop_last=False,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(
            list(self._classifier.parameters()) + list(self._summary_net.parameters()),
            lr=learning_rate,
        )

        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        # If we're retraining from scratch each round, reset the neural posterior
        # to the untrained copy we made at the start.
        if self._retrain_from_scratch_each_round:
            self._classifier = deepcopy(self._classifier)

        def _get_log_prob(parameters, observations):

            # num_atoms = parameters.shape[0]
            num_atoms = self._num_atoms if self._num_atoms > 0 else batch_size

            repeated_observations = utils.repeat_rows(observations, num_atoms)

            # Choose between 1 and num_atoms - 1 parameters from the rest
            # of the batch for each observation.
            assert 0 < num_atoms - 1 < batch_size
            probs = (
                (1 / (batch_size - 1))
                * torch.ones(batch_size, batch_size)
                * (1 - torch.eye(batch_size))
            )
            choices = torch.multinomial(
                probs, num_samples=num_atoms - 1, replacement=False
            )
            contrasting_parameters = parameters[choices]

            atomic_parameters = torch.cat(
                (parameters[:, None, :], contrasting_parameters), dim=1
            ).reshape(batch_size * num_atoms, -1)

            inputs = torch.cat((atomic_parameters, repeated_observations), dim=1)

            logits = self._classifier(inputs).reshape(batch_size, num_atoms)

            log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)

            return log_prob

        epochs = 0
        while True:

            # Train for a single epoch.
            self._classifier.train()
            for parameters, observations in train_loader:
                optimizer.zero_grad()
                log_prob = _get_log_prob(parameters, observations)
                loss = -torch.mean(log_prob)
                loss.backward()
                optimizer.step()

            epochs += 1

            # calculate validation performance
            self._classifier.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for parameters, observations in val_loader:
                    log_prob = _get_log_prob(parameters, observations)
                    log_prob_sum += log_prob.sum().item()
                validation_log_prob = log_prob_sum / num_validation_examples

            # check for improvement
            if validation_log_prob > best_validation_log_prob:
                best_model_state_dict = deepcopy(self._classifier.state_dict())
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

            # if no validation improvement over many epochs, stop training
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self._classifier.load_state_dict(best_model_state_dict)
                break

        # Update summary.
        self._summary["epochs"].append(epochs)
        self._summary["best-validation-log-probs"].append(best_validation_log_prob)

    @property
    def summary(self):
        return self._summary

    def _summarize(self, round_):

        # Update summaries.
        try:
            mmd = utils.unbiased_mmd_squared(
                self._parameter_bank[-1],
                self._simulator.get_ground_truth_posterior_samples(num_samples=1000),
            )
            self._summary["mmds"].append(mmd.item())
        except:
            pass

        median_observation_distance = torch.median(
            torch.sqrt(
                torch.sum(
                    (self._observation_bank[-1] - self._true_observation.reshape(1, -1))
                    ** 2,
                    dim=-1,
                )
            )
        )
        self._summary["median-observation-distances"].append(
            median_observation_distance.item()
        )

        negative_log_prob_true_parameters = -utils.gaussian_kde_log_eval(
            samples=self._parameter_bank[-1],
            query=self._simulator.get_ground_truth_parameters().reshape(1, -1),
        )
        self._summary["negative-log-probs-true-parameters"].append(
            negative_log_prob_true_parameters.item()
        )

        # Plot most recently sampled parameters in TensorBoard.
        parameters = utils.tensor2numpy(self._parameter_bank[-1])
        figure = utils.plot_hist_marginals(
            data=parameters,
            ground_truth=utils.tensor2numpy(
                self._simulator.get_ground_truth_parameters()
            ).reshape(-1),
            lims=self._simulator.parameter_plotting_limits,
        )
        self._summary_writer.add_figure(
            tag="posterior-samples", figure=figure, global_step=round_ + 1
        )

        self._summary_writer.add_scalar(
            tag="epochs-trained",
            scalar_value=self._summary["epochs"][-1],
            global_step=round_ + 1,
        )

        self._summary_writer.add_scalar(
            tag="best-validation-log-prob",
            scalar_value=self._summary["best-validation-log-probs"][-1],
            global_step=round_ + 1,
        )

        self._summary_writer.add_scalar(
            tag="median-observation-distance",
            scalar_value=self._summary["median-observation-distances"][-1],
            global_step=round_ + 1,
        )

        self._summary_writer.add_scalar(
            tag="negative-log-prob-true-parameters",
            scalar_value=self._summary["negative-log-probs-true-parameters"][-1],
            global_step=round_ + 1,
        )

        if self._summary["mmds"]:
            self._summary_writer.add_scalar(
                tag="mmd",
                scalar_value=self._summary["mmds"][-1],
                global_step=round_ + 1,
            )

        self._summary_writer.flush()


class NeuralPotentialFunction:
    """
    Implementation of a potential function for Pyro MCMC which uses a binary classifier
    to evaluate a quantity proportional to the likelihood.
    """

    def __init__(self, classifier, prior, true_observation):
        """
        :param neural_likelihood: Binary classifier which has learned an approximation
        to the likelihood up to a constant.
        :param prior: Distribution object with 'log_prob' method.
        :param true_observation: torch.Tensor containing true observation x0.
        """

        self.classifier = classifier
        self.prior = prior
        self.true_observation = true_observation

    def __call__(self, parameters_dict):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        :param inputs_dict: dict of parameter values which need evaluation for MCMC.
        :return: torch.Tensor potential ~ -[log r(x0, theta) + log p(theta)]
        """

        parameters = next(iter(parameters_dict.values()))
        log_ratio = self.classifier(
            torch.cat((parameters, self.true_observation)).reshape(1, -1)
        )

        # If prior is uniform we need to sum across last dimension.
        if isinstance(self.prior, distributions.Uniform):
            potential = -(log_ratio + self.prior.log_prob(parameters).sum(-1))
        else:
            potential = -(log_ratio + self.prior.log_prob(parameters))

        return potential


def test_():
    task = "lotka-volterra"
    simulator, prior = simulators.get_simulator_and_prior(task)
    parameter_dim, observation_dim = (
        simulator.parameter_dim,
        simulator.observation_dim,
    )
    true_observation = simulator.get_ground_truth_observation()

    classifier = utils.get_classifier("mlp", parameter_dim, observation_dim)
    ratio_estimator = SRE(
        simulator=simulator,
        true_observation=true_observation,
        classifier=classifier,
        prior=prior,
        num_atoms=-1,
        mcmc_method="slice-np",
        retrain_from_scratch_each_round=False,
    )

    num_rounds, num_simulations_per_round = 10, 1000
    ratio_estimator.run_inference(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    samples = ratio_estimator.sample_posterior(num_samples=2500)
    samples = utils.tensor2numpy(samples)
    figure = utils.plot_hist_marginals(
        data=samples,
        ground_truth=utils.tensor2numpy(
            simulator.get_ground_truth_parameters()
        ).reshape(-1),
        lims=[-4, 4],
    )
    figure.savefig(os.path.join(utils.get_output_root(), "corner-posterior-ratio.pdf"))

    mmds = ratio_estimator.summary["mmds"]
    if mmds:
        figure, axes = plt.subplots(1, 1)
        axes.plot(
            np.arange(
                0, num_rounds * num_simulations_per_round, num_simulations_per_round
            ),
            np.array(mmds),
            "-o",
            linewidth=2,
        )
        figure.savefig(os.path.join(utils.get_output_root(), "mmd-ratio.pdf"))


def main():
    test_()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

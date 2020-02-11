import numpy as np
import os
import torch

import sbi.simulators as simulators
import pyknos.utils as utils

from copy import deepcopy
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import distributions, multiprocessing as mp, nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sbi.mcmc import Slice, SliceSampler

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    # input("CUDA not available, do you wish to continue?")
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")


class APT:
    """
    Implementation of
    'Automatic Posterior Transformation for Likelihood-free Inference'
    Greenberg et al.
    ICML 2019
    https://arxiv.org/abs/1905.07488
    """

    def __init__(
        self,
        simulator,
        prior,
        true_observation,
        neural_posterior,
        num_atoms=-1,
        use_combined_loss=False,
        train_with_mcmc=False,
        mcmc_method="slice-np",
        summary_net=None,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        summary_writer=None,
    ):
        """
        :param simulator:
            Python object with 'simulate' method which takes a torch.Tensor
            of parameter values, and returns a simulation result for each parameter
            as a torch.Tensor.
        :param prior: Distribution
            Distribution object with 'log_prob' and 'sample' methods.
        :param true_observation: torch.Tensor [observation_dim] or [1, observation_dim]
            True observation x0 for which to perform inference on the posterior p(theta | x0).
        :param neural_posterior: nets.Module
            Conditional density estimator q(theta | x) with 'log_prob' and 'sample' methods.
        :param num_atoms: int
            Number of atoms to use for classification.
            If -1, use all other parameters in minibatch.
        :param use_combined_loss: bool
            Whether to jointly train prior samples using maximum likelihood.
            Useful to prevent density leaking when using box uniform priors.
        :param train_with_mcmc: bool
            Whether to sample using MCMC instead of i.i.d. sampling at the end of each round
        :param mcmc_method: str
            MCMC method to use if 'train_with_mcmc' is True.
            One of ['slice-numpy', 'hmc', 'nuts'].
        :param summary_net: nets.Module
            Optional network which may be used to produce feature vectors
            f(x) for high-dimensional observations.
        :param retrain_from_scratch_each_round: bool
            Whether to retrain the conditional density estimator for the posterior
            from scratch each round.
        :param discard_prior_samples: bool
            Whether to discard prior samples from round two onwards.
        :param summary_writer: SummaryWriter
            Optionally pass summary writer.
            If None, will create one internally.
        """

        self._simulator = simulator
        self._prior = prior
        self._true_observation = true_observation
        self._neural_posterior = neural_posterior

        assert isinstance(num_atoms, int), "Number of atoms must be an integer."
        self._num_atoms = num_atoms

        self._use_combined_loss = use_combined_loss

        # We may want to summarize high-dimensional observations.
        # This may be either a fixed or learned transformation.
        if summary_net is None:
            self._summary_net = nn.Identity()
        else:
            self._summary_net = summary_net

        self._mcmc_method = mcmc_method
        self._train_with_mcmc = train_with_mcmc

        # HMC and NUTS from Pyro.
        # Defining the potential function as an object means Pyro's MCMC scheme
        # can pickle it to be used across multiple chains in parallel, even if
        # the potential function requires evaluating a neural likelihood as is the
        # case here.
        self._potential_function = NeuralPotentialFunction(
            neural_posterior, prior, self._true_observation
        )

        # Axis-aligned slice sampling implementation in NumPy
        target_log_prob = (
            lambda parameters: self._neural_posterior.log_prob(
                inputs=torch.Tensor(parameters).reshape(1, -1),
                context=self._true_observation.reshape(1, -1),
            ).item()
            if not np.isinf(self._prior.log_prob(torch.Tensor(parameters)).sum().item())
            else -np.inf
        )
        self._neural_posterior.eval()
        self.posterior_sampler = SliceSampler(
            utils.tensor2numpy(self._prior.sample((1,))).reshape(-1),
            lp_f=target_log_prob,
            thin=10,
        )
        self._neural_posterior.train()

        self._retrain_from_scratch_each_round = retrain_from_scratch_each_round
        # If we're retraining from scratch each round,
        # keep a copy of the original untrained model for reinitialization.
        self._untrained_neural_posterior = deepcopy(neural_posterior)

        self._discard_prior_samples = discard_prior_samples

        # Need somewhere to store (parameter, observation) pairs from each round.
        self._parameter_bank, self._observation_bank, self._prior_masks = [], [], []

        self._model_bank = []

        self._total_num_generated_examples = 0

        # Each APT run has an associated log directory for TensorBoard output.
        if summary_writer is None:
            log_dir = os.path.join(
                utils.get_log_root(), "apt", simulator.name, utils.get_timestamp()
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
            "epochs": [],
            "best-validation-log-probs": [],
            "rejection-sampling-acceptance-rates": [],
        }

    def run_inference(self, num_rounds, num_simulations_per_round):
        """
        This runs APT for num_rounds rounds, using num_simulations_per_round calls to
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
                    parameter_sample_fn=lambda num_samples: self.sample_posterior_mcmc(
                        num_samples
                    )
                    if self._train_with_mcmc
                    else self.sample_posterior(num_samples),
                    num_samples=num_simulations_per_round,
                )

            # Store (parameter, observation) pairs.
            self._parameter_bank.append(torch.Tensor(parameters))
            self._observation_bank.append(torch.Tensor(observations))
            self._prior_masks.append(
                torch.ones(num_simulations_per_round, 1)
                if round_ == 0
                else torch.zeros(num_simulations_per_round, 1)
            )

            # Fit posterior using newly aggregated data set.
            self._fit_posterior(round_=round_)

            # Store models at end of each round.
            self._model_bank.append(deepcopy(self._neural_posterior))
            self._model_bank[-1].eval()

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

    def sample_posterior(self, num_samples, true_observation=None):
        """
        Samples from posterior for true observation q(theta | x0) using most recent
        posterior estimate.

        :param num_samples: int
            Number of samples to generate.
        :param true_observation: torch.Tensor [observation_dim] or [1, observation_dim]
            Optionally pass true observation for inference.
            Otherwise uses true observation given at instantiation.
        :return: torch.Tensor [num_samples, parameter_dim]
            Posterior parameter samples.
        """

        true_observation = (
            true_observation if true_observation is not None else self._true_observation
        )

        # Always sample in eval mode.
        self._neural_posterior.eval()

        # Rejection sampling is potentially needed for the posterior.
        # This is because the prior may not have support everywhere.
        # The posterior may also be constrained to the same support,
        # but we don't know this a priori.
        samples = []
        num_remaining_samples = num_samples
        total_num_accepted, self._total_num_generated_examples = 0, 0
        while num_remaining_samples > 0:

            # Generate samples from posterior.
            candidate_samples = self._neural_posterior.sample(
                max(10000, num_samples), context=true_observation.reshape(1, -1)
            ).squeeze(0)

            # Evaluate posterior samples under the prior.
            prior_log_prob = self._prior.log_prob(candidate_samples)
            if isinstance(self._prior, distributions.Uniform):
                prior_log_prob = prior_log_prob.sum(-1)

            # Keep those samples which have non-zero probability under the prior.
            accepted_samples = candidate_samples[~torch.isinf(prior_log_prob)]
            samples.append(accepted_samples.detach())

            # Update remaining number of samples needed.
            num_accepted = (~torch.isinf(prior_log_prob)).sum().item()
            num_remaining_samples -= num_accepted
            total_num_accepted += num_accepted

            # Keep track of acceptance rate
            self._total_num_generated_examples += candidate_samples.shape[0]

        # Back to training mode.
        self._neural_posterior.train()

        # Aggregate collected samples.
        samples = torch.cat(samples)

        # Make sure we have the right amount.
        samples = samples[:num_samples, ...]
        assert samples.shape[0] == num_samples

        return samples

    def sample_posterior_mcmc(self, num_samples, thin=10):
        """
        Samples from posterior for true observation q(theta | x0) using MCMC.

        :param num_samples: Number of samples to generate.
        :param mcmc_method: Which MCMC method to use ['metropolis-hastings', 'slice', 'hmc', 'nuts']
        :param thin: Generate (num_samples * thin) samples in total, then select every
        'thin' sample.
        :return: torch.Tensor of shape [num_samples, parameter_dim]
        """

        # Always sample in eval mode.
        self._neural_posterior.eval()

        if self._mcmc_method == "slice-np":
            self.posterior_sampler.gen(20)  # Burn-in for 200 samples
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
        self._neural_posterior.train()

        return samples

    def _fit_posterior(
        self,
        round_,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        clip_grad_norm=True,
    ):
        """
        Trains the conditional density estimator for the posterior by maximizing the
        proposal posterior using the most recently aggregated bank of (parameter, observation)
        pairs.
        Uses early stopping on a held-out validation set as a terminating condition.

        :param round_: int
            Which round we're currently in. Needed when sampling procedure is
            not simply sampling from (proposal) marginal.
        :param batch_size: int
            Size of batch to use for training.
        :param learning_rate: float
            Learning rate for Adam optimizer.
        :param validation_fraction: float in [0, 1]
            The fraction of data to use for validation.
        :param stop_after_epochs: int
            The number of epochs to wait for improvement on the
            validation set before terminating training.
        :param clip_grad_norm: bool
            Whether to clip norm of gradients or not.
        :return: None
        """

        if self._discard_prior_samples and (round_ > 0):
            ix = 1
        else:
            ix = 0

        # Get total number of training examples.
        num_examples = torch.cat(self._parameter_bank[ix:]).shape[0]

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
            torch.cat(self._parameter_bank[ix:]),
            torch.cat(self._observation_bank[ix:]),
            torch.cat(self._prior_masks[ix:]),
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
            list(self._neural_posterior.parameters())
            + list(self._summary_net.parameters()),
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
        if self._retrain_from_scratch_each_round and round_ > 0:
            # self._neural_posterior = deepcopy(self._untrained_neural_posterior)
            self._neural_posterior = deepcopy(self._model_bank[0])

        def _get_log_prob_proposal_posterior(inputs, context, masks):
            """
            We have two main options when evaluating the proposal posterior.
            (1) Generate atoms from the proposal prior.
            (2) Generate atoms from a more targeted distribution,
            such as the most recent posterior.
            If we choose the latter, it is likely beneficial not to do this in the first
            round, since we would be sampling from a randomly initialized neural density
            estimator.

            :param inputs: torch.Tensor Batch of parameters.
            :param context: torch.Tensor Batch of observations.
            :return: torch.Tensor [1] log_prob_proposal_posterior
            """

            log_prob_posterior_non_atomic = self._neural_posterior.log_prob(
                inputs, context
            )

            # just do maximum likelihood in the first round
            if round_ == 0:
                return log_prob_posterior_non_atomic

            num_atoms = self._num_atoms if self._num_atoms > 0 else batch_size

            # Each set of parameter atoms is evaluated using the same observation,
            # so we repeat rows of the context.
            # e.g. [1, 2] -> [1, 1, 2, 2]
            repeated_context = utils.repeat_rows(context, num_atoms)

            # To generate the full set of atoms for a given item in the batch,
            # we sample without replacement num_atoms - 1 times from the rest
            # of the parameters in the batch.
            assert 0 < num_atoms - 1 < batch_size
            probs = (
                (1 / (batch_size - 1))
                * torch.ones(batch_size, batch_size)
                * (1 - torch.eye(batch_size))
            )
            choices = torch.multinomial(
                probs, num_samples=num_atoms - 1, replacement=False
            )
            contrasting_inputs = inputs[choices]

            # We can now create our sets of atoms from the contrasting parameter sets
            # we have generated.
            atomic_inputs = torch.cat(
                (inputs[:, None, :], contrasting_inputs), dim=1
            ).reshape(batch_size * num_atoms, -1)

            # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
            log_prob_posterior = self._neural_posterior.log_prob(
                atomic_inputs, repeated_context
            )
            assert utils.notinfnotnan(
                log_prob_posterior
            ), "NaN/inf detected in posterior eval."
            log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

            # Get (batch_size * num_atoms) log prob prior evals.
            if isinstance(self._prior, distributions.Uniform):
                log_prob_prior = self._prior.log_prob(atomic_inputs).sum(-1)
                # log_prob_prior = torch.zeros(log_prob_prior.shape)
            else:
                log_prob_prior = self._prior.log_prob(atomic_inputs)
            log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
            assert utils.notinfnotnan(log_prob_prior), "NaN/inf detected in prior eval."

            # Compute unnormalized proposal posterior.
            unnormalized_log_prob_proposal_posterior = (
                log_prob_posterior - log_prob_prior
            )

            # Normalize proposal posterior across discrete set of atoms.
            log_prob_proposal_posterior = unnormalized_log_prob_proposal_posterior[
                :, 0
            ] - torch.logsumexp(unnormalized_log_prob_proposal_posterior, dim=-1)
            assert utils.notinfnotnan(
                log_prob_proposal_posterior
            ), "NaN/inf detected in proposal posterior eval."

            if self._use_combined_loss:
                masks = masks.reshape(-1)

                log_prob_proposal_posterior = (
                    masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
                )

            return log_prob_proposal_posterior

        epochs = 0
        while True:

            # Train for a single epoch.
            self._neural_posterior.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, context, masks = (
                    batch[0].to(device),
                    batch[1].to(device),
                    batch[2].to(device),
                )
                summarized_context = self._summary_net(context)
                log_prob_proposal_posterior = _get_log_prob_proposal_posterior(
                    inputs, summarized_context, masks
                )
                loss = -torch.mean(log_prob_proposal_posterior)
                loss.backward()
                if clip_grad_norm:
                    clip_grad_norm_(self._neural_posterior.parameters(), max_norm=5.0)
                optimizer.step()

            epochs += 1

            # Calculate validation performance.
            self._neural_posterior.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, context, masks = (
                        batch[0].to(device),
                        batch[1].to(device),
                        batch[2].to(device),
                    )
                    log_prob = _get_log_prob_proposal_posterior(inputs, context, masks)
                    log_prob_sum += log_prob.sum().item()
            validation_log_prob = log_prob_sum / num_validation_examples

            # Check for improvement in validation performance over previous epochs.
            if validation_log_prob > best_validation_log_prob:
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
                best_model_state_dict = deepcopy(self._neural_posterior.state_dict())
            else:
                epochs_since_last_improvement += 1

            # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self._neural_posterior.load_state_dict(best_model_state_dict)
                break

        # Update summary.
        self._summary["epochs"].append(epochs)
        self._summary["best-validation-log-probs"].append(best_validation_log_prob)

    def _estimate_acceptance_rate(self, num_samples=int(1e5), true_observation=None):
        """
        Estimates rejection sampling acceptance rates.

        :param num_samples:
            Number of samples to use.
        :param true_observation:
            Observation on which to condition.
            If None, use true observation given at initialization.
        :return: float in [0, 1]
            Fraction of accepted samples.
        """
        true_observation = (
            true_observation if true_observation is not None else self._true_observation
        )

        # Always sample in eval mode.
        self._neural_posterior.eval()

        total_num_accepted_samples, total_num_generated_samples = 0, 0
        while total_num_generated_samples < num_samples:

            # Generate samples from posterior.
            candidate_samples = self._neural_posterior.sample(
                10000, context=true_observation.reshape(1, -1)
            ).squeeze(0)

            # Evaluate posterior samples under the prior.
            prior_log_prob = self._prior.log_prob(candidate_samples)
            if isinstance(self._prior, distributions.Uniform):
                prior_log_prob = prior_log_prob.sum(-1)

            # Update remaining number of samples needed.
            num_accepted_samples = (~torch.isinf(prior_log_prob)).sum().item()
            total_num_accepted_samples += num_accepted_samples

            # Keep track of acceptance rate
            total_num_generated_samples += candidate_samples.shape[0]

        # Back to training mode.
        self._neural_posterior.train()

        return total_num_accepted_samples / total_num_generated_samples

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

        # Median |x - x0| for most recent round.
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

        # KDE estimate of negative log prob true parameters using
        # parameters from most recent round.
        negative_log_prob_true_parameters = -utils.gaussian_kde_log_eval(
            samples=self._parameter_bank[-1],
            query=self._simulator.get_ground_truth_parameters().reshape(1, -1),
        )
        self._summary["negative-log-probs-true-parameters"].append(
            negative_log_prob_true_parameters.item()
        )

        # Rejection sampling acceptance rate
        rejection_sampling_acceptance_rate = self._estimate_acceptance_rate()
        self._summary["rejection-sampling-acceptance-rates"].append(
            rejection_sampling_acceptance_rate
        )

        # Plot most recently sampled parameters.
        parameters = utils.tensor2numpy(self._parameter_bank[-1])
        figure = utils.plot_hist_marginals(
            data=parameters,
            ground_truth=utils.tensor2numpy(
                self._simulator.get_ground_truth_parameters()
            ).reshape(-1),
            lims=self._simulator.parameter_plotting_limits,
        )

        # Write quantities using SummaryWriter.
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

        self._summary_writer.add_scalar(
            tag="rejection-sampling-acceptance-rate",
            scalar_value=self._summary["rejection-sampling-acceptance-rates"][-1],
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
    Implementation of a potential function for Pyro MCMC which uses a classifier
    to evaluate a quantity proportional to the likelihood.
    """

    def __init__(self, posterior, prior, true_observation):
        """
        :param neural_likelihood: Binary classifier which has learned an approximation
        to the likelihood up to a constant.
        :param prior: Distribution object with 'log_prob' method.
        :param true_observation: torch.Tensor containing true observation x0.
        """

        self.prior = prior
        self.posterior = posterior
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
        potential = -self.posterior.log_prob(
            inputs=parameters, context=self.true_observation
        )
        if isinstance(self.prior, distributions.Uniform):
            log_prob_prior = self.prior.log_prob(parameters).sum(-1)
        else:
            log_prob_prior = self.prior.log_prob(parameters)
        log_prob_prior[~torch.isinf(log_prob_prior)] = 1
        potential *= log_prob_prior

        return potential


def test_():
    task = "nonlinear-gaussian"
    simulator, prior = simulators.get_simulator_and_prior(task)
    parameter_dim, observation_dim = (
        simulator.parameter_dim,
        simulator.observation_dim,
    )
    true_observation = simulator.get_ground_truth_observation()
    neural_posterior = utils.get_neural_posterior(
        "maf", parameter_dim, observation_dim, simulator
    )
    apt = APT(
        simulator=simulator,
        true_observation=true_observation,
        prior=prior,
        neural_posterior=neural_posterior,
        num_atoms=-1,
        use_combined_loss=False,
        train_with_mcmc=False,
        mcmc_method="slice-np",
        summary_net=None,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
    )

    num_rounds, num_simulations_per_round = 20, 1000
    apt.run_inference(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    samples = apt.sample_posterior(2500)
    samples = utils.tensor2numpy(samples)
    figure = utils.plot_hist_marginals(
        data=samples,
        ground_truth=utils.tensor2numpy(
            simulator.get_ground_truth_parameters()
        ).reshape(-1),
        lims=simulator.parameter_plotting_limits,
    )
    figure.savefig(os.path.join(utils.get_output_root(), "corner-posterior-apt.pdf"))

    samples = apt.sample_posterior_mcmc(num_samples=1000)
    samples = utils.tensor2numpy(samples)
    figure = utils.plot_hist_marginals(
        data=samples,
        ground_truth=utils.tensor2numpy(
            simulator.get_ground_truth_parameters()
        ).reshape(-1),
        lims=simulator.parameter_plotting_limits,
    )
    figure.savefig(
        os.path.join(utils.get_output_root(), "corner-posterior-apt-mcmc.pdf")
    )


def main():
    test_()


if __name__ == "__main__":
    main()

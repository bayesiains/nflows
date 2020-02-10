import os
import torch

import lfi.simulators as simulators
import lfi.utils as utils

from copy import deepcopy
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import distributions, multiprocessing as mp, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lfi.mcmc import Slice, SliceSampler

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    #input("CUDA not available, do you wish to continue?")
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")


class SNL:
    """
    Implementation of
    'Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows'
    Papamakarios et al.
    AISTATS 2019
    https://arxiv.org/abs/1805.07226
    """

    def __init__(
        self,
        simulator,
        prior,
        true_observation,
        neural_likelihood,
        mcmc_method="slice-np",
        summary_writer=None,
    ):
        """

        :param simulator: Python object with 'simulate' method which takes a torch.Tensor
        of parameter values, and returns a simulation result for each parameter as a torch.Tensor.
        :param prior: Distribution object with 'log_prob' and 'sample' methods.
        :param true_observation: torch.Tensor containing the observation x0 for which to
        perform inference on the posterior p(theta | x0).
        :param neural_likelihood: Conditional density estimator q(x | theta) in the form of an
        nets.Module. Must have 'log_prob' and 'sample' methods.
        :param mcmc_method: MCMC method to use for posterior sampling. Must be one of
        ['slice', 'hmc', 'nuts'].
        """

        self._simulator = simulator
        self._prior = prior
        self._true_observation = true_observation
        self._neural_likelihood = neural_likelihood
        self._mcmc_method = mcmc_method

        # Defining the potential function as an object means Pyro's MCMC scheme
        # can pickle it to be used across multiple chains in parallel, even if
        # the potential function requires evaluating a neural likelihood as is the
        # case here.
        self._potential_function = NeuralPotentialFunction(
            neural_likelihood=self._neural_likelihood,
            prior=self._prior,
            true_observation=self._true_observation,
        )

        # TODO: decide on Slice Sampling implementation
        target_log_prob = (
            lambda parameters: self._neural_likelihood.log_prob(
                inputs=self._true_observation.reshape(1, -1),
                context=torch.Tensor(parameters).reshape(1, -1),
            ).item()
            + self._prior.log_prob(torch.Tensor(parameters)).sum().item()
        )
        self._neural_likelihood.eval()
        self.posterior_sampler = SliceSampler(
            utils.tensor2numpy(self._prior.sample((1,))).reshape(-1),
            lp_f=target_log_prob,
            thin=10,
        )
        self._neural_likelihood.train()

        # Need somewhere to store (parameter, observation) pairs from each round.
        self._parameter_bank, self._observation_bank = [], []

        # Each SNL run has an associated log directory for TensorBoard output.
        if summary_writer is None:
            log_dir = os.path.join(
                utils.get_log_root(), "snl", simulator.name, utils.get_timestamp()
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
        This runs SNL for num_rounds rounds, using num_simulations_per_round calls to
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

            # Fit neural likelihood to newly aggregated dataset.
            self._fit_likelihood()

            # Update description for progress bar.
            round_description = (
                f"-------------------------\n"
                f"||||| ROUND {round_ + 1} STATS |||||:\n"
                f"-------------------------\n"
                f"Epochs trained: {self._summary['epochs'][-1]}\n"
                f"Best validation performance: {self._summary['best-validation-log-probs'][-1]:.4f}\n\n"
            )

            # Update TensorBoard and summary dict.
            self._summarize(round_)

    def sample_posterior(self, num_samples, thin=1):
        """
        Samples from posterior for true observation q(theta | x0) ~ q(x0 | theta) p(theta)
        using most recent likelihood estimate q(x0 | theta) with MCMC.

        :param num_samples: Number of samples to generate.
        :param thin: Generate (num_samples * thin) samples in total, then select every
        'thin' sample.
        :return: torch.Tensor of shape [num_samples, parameter_dim]
        """

        # Always sample in eval mode.
        self._neural_likelihood.eval()

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

            # TODO: decide on way to initialize chain
            initial_params = self._prior.sample((num_chains,))
            sampler = MCMC(
                kernel=kernel,
                num_samples=num_samples // num_chains + num_chains,
                warmup_steps=200,
                initial_params={"": initial_params},
                num_chains=num_chains,
            )
            sampler.run()
            samples = next(iter(sampler.get_samples().values())).reshape(
                -1, self._simulator.parameter_dim
            )

            samples = samples[:num_samples].to(device)
            assert samples.shape[0] == num_samples

        # Back to training mode.
        self._neural_likelihood.train()

        return samples

    def _fit_likelihood(
        self,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
    ):
        """
        Trains the conditional density estimator for the likelihood by maximum likelihood
        on the most recently aggregated bank of (parameter, observation) pairs.
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
            torch.cat(self._observation_bank), torch.cat(self._parameter_bank)
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

        optimizer = optim.Adam(self._neural_likelihood.parameters(), lr=learning_rate)
        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        epochs = 0
        while True:

            # Train for a single epoch.
            self._neural_likelihood.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, context = batch[0].to(device), batch[1].to(device)
                log_prob = self._neural_likelihood.log_prob(inputs, context=context)
                loss = -torch.mean(log_prob)
                loss.backward()
                clip_grad_norm_(self._neural_likelihood.parameters(), max_norm=5.0)
                optimizer.step()

            epochs += 1

            # Calculate validation performance.
            self._neural_likelihood.eval()
            log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, context = batch[0].to(device), batch[1].to(device)
                    log_prob = self._neural_likelihood.log_prob(inputs, context=context)
                    log_prob_sum += log_prob.sum().item()
            validation_log_prob = log_prob_sum / num_validation_examples

            # Check for improvement in validation performance over previous epochs.
            if validation_log_prob > best_validation_log_prob:
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
                best_model_state_dict = deepcopy(self._neural_likelihood.state_dict())
            else:
                epochs_since_last_improvement += 1

            # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self._neural_likelihood.load_state_dict(best_model_state_dict)
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
            print(mmd.item())
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
            tag="best-validation-log-prob",
            scalar_value=self._summary["best-validation-log-probs"][-1],
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
    Implementation of a potential function for Pyro MCMC which uses a neural density
    estimator to evaluate the likelihood.
    """

    def __init__(self, neural_likelihood, prior, true_observation):
        """
        :param neural_likelihood: Conditional density estimator with 'log_prob' method.
        :param prior: Distribution object with 'log_prob' method.
        :param true_observation: torch.Tensor containing true observation x0.
        """

        self._neural_likelihood = neural_likelihood
        self._prior = prior
        self._true_observation = true_observation

    def __call__(self, inputs_dict):
        """
        Call method allows the object to be used as a function.
        Evaluates the given parameters using a given neural likelhood, prior,
        and true observation.

        :param inputs_dict: dict of parameter values which need evaluation for MCMC.
        :return: torch.Tensor potential ~ -[log q(x0 | theta) + log p(theta)]
        """

        parameters = next(iter(inputs_dict.values()))
        log_likelihood = self._neural_likelihood.log_prob(
            inputs=self._true_observation.reshape(1, -1).to("cpu"),
            context=parameters.reshape(1, -1),
        )

        # If prior is uniform we need to sum across last dimension.
        if isinstance(self._prior, distributions.Uniform):
            potential = -(log_likelihood + self._prior.log_prob(parameters).sum(-1))
        else:
            potential = -(log_likelihood + self._prior.log_prob(parameters))

        return potential


def main():
    task = "mg1"
    simulator, prior = simulators.get_simulator_and_prior(task)
    parameter_dim, observation_dim = (
        simulator.parameter_dim,
        simulator.observation_dim,
    )
    true_observation = simulator.get_ground_truth_observation()
    neural_likelihood = utils.get_neural_likelihood(
        "maf", parameter_dim, observation_dim
    )
    snl = SNL(
        simulator=simulator,
        true_observation=true_observation,
        prior=prior,
        neural_likelihood=neural_likelihood,
        mcmc_method="slice-np",
    )

    num_rounds, num_simulations_per_round = 10, 1000
    snl.run_inference(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    samples = snl.sample_posterior(1000)
    samples = utils.tensor2numpy(samples)
    figure = utils.plot_hist_marginals(
        data=samples,
        ground_truth=utils.tensor2numpy(
            simulator.get_ground_truth_parameters()
        ).reshape(-1),
        lims=simulator.parameter_plotting_limits,
    )
    figure.savefig("./corner-posterior-snl.pdf")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

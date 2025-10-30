"""
Parallel Tempering MCMC with Multi-GPU Support and W&B Logging
===============================================================
This script implements parallel tempering (replica exchange MCMC) for Bayesian
neural network inference across multiple GPUs using PyTorch and DeepSpeed.
All metrics and diagnostics are logged to Weights & Biases.

Each GPU runs a Markov chain at a different temperature, with periodic swaps
between chains to improve mixing and exploration of multimodal posteriors.

Author: [Your Name]
Date: 2025-10-29
"""

import argparse
import logging
import os
from typing import Tuple, Dict, List, Optional
import json
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import deepspeed
import wandb


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Embedded DeepSpeed Configuration
DEEPSPEED_CONFIG = {
    "train_batch_size": 100,
    "train_micro_batch_size_per_gpu": 100,
    "steps_per_print": 100,
    "wall_clock_breakdown": False,
    "dump_state": False,
    "gradient_accumulation_steps": 1,
    "prescale_gradients": False,
    "bf16": {
        "enabled": False
    },
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 0
    },
    "communication_data_type": "fp32",
    "distributed_backend": "nccl"
}


class SyntheticDataset(Dataset):
    """
    Synthetic regression dataset for Bayesian inference demonstration.

    Generates noisy observations from a true underlying function.
    """

    def __init__(self, num_samples: int = 1000, input_dim: int = 10,
                 noise_std: float = 0.1) -> None:
        """
        Initialize synthetic dataset.

        Args:
            num_samples: Number of samples to generate
            input_dim: Dimension of input features
            noise_std: Standard deviation of observation noise
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.noise_std = noise_std

        # Generate fixed data for consistency across replicas
        torch.manual_seed(42)
        self.X = torch.randn(num_samples, input_dim)

        # True function: y = X @ true_weights + noise
        self.true_weights = torch.randn(input_dim, 1)
        self.y = self.X @ self.true_weights + noise_std * torch.randn(num_samples, 1)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (input_tensor, target)
        """
        return self.X[idx], self.y[idx]


class BayesianMLP(nn.Module):
    """
    Multi-layer perceptron for Bayesian inference.

    Simple architecture suitable for demonstrating parallel tempering MCMC.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 50,
                 output_dim: int = 1) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super(BayesianMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ParallelTemperingMCMC:
    """
    Parallel Tempering MCMC sampler for Bayesian neural networks.

    Each GPU runs a replica at different temperature. Higher temperatures
    allow easier exploration, while temperature=1.0 samples from true posterior.
    """

    def __init__(self, model: nn.Module, temperatures: List[float],
                 prior_std: float = 1.0, noise_std: float = 0.1,
                 local_rank: int = 0) -> None:
        """
        Initialize parallel tempering sampler.

        Args:
            model: Neural network model
            temperatures: List of temperatures for each replica
            prior_std: Standard deviation of Gaussian prior on weights
            noise_std: Standard deviation of observation noise
            local_rank: GPU rank for this replica
        """
        self.model = model
        self.temperatures = temperatures
        self.temperature = temperatures[local_rank]
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.local_rank = local_rank

        # Statistics tracking
        self.samples = []
        self.log_posteriors = []
        self.log_priors = []
        self.log_likelihoods = []
        self.acceptance_count = 0
        self.proposal_count = 0
        self.swap_acceptance_count = 0
        self.swap_proposal_count = 0

        # Track parameter statistics over time
        self.param_means = []
        self.param_stds = []

        logger.info(f"Replica {local_rank} initialized with temperature {self.temperature:.3f}")

    def log_prior(self) -> torch.Tensor:
        """
        Compute log prior probability of current model parameters.

        Assumes Gaussian prior: p(θ) ~ N(0, prior_std^2 * I)

        Returns:
            Log prior probability
        """
        log_prob = 0.0
        for param in self.model.parameters():
            log_prob += torch.sum(-0.5 * (param / self.prior_std) ** 2)
        return log_prob

    def log_likelihood(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Compute log likelihood of data given current model parameters.

        Assumes Gaussian likelihood: p(y|X,θ) ~ N(f(X;θ), noise_std^2)

        Args:
            data_loader: DataLoader containing observations

        Returns:
            Log likelihood
        """
        self.model.eval()
        log_lik = 0.0

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.local_rank)
                y = y.to(self.local_rank)

                predictions = self.model(X)
                residuals = y - predictions
                log_lik += torch.sum(-0.5 * (residuals / self.noise_std) ** 2)

        return log_lik

    def log_posterior(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log posterior probability (up to constant).

        log p(θ|D) ∝ log p(D|θ) + log p(θ)

        Args:
            data_loader: DataLoader containing observations

        Returns:
            Tuple of (log_posterior, log_likelihood, log_prior)
        """
        log_prior = self.log_prior()
        log_lik = self.log_likelihood(data_loader)
        log_post = log_lik + log_prior
        return log_post, log_lik, log_prior

    def propose_parameters(self, proposal_std: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Propose new parameters using Gaussian random walk.

        θ' = θ + ε, where ε ~ N(0, proposal_std^2 * I)

        Args:
            proposal_std: Standard deviation of proposal distribution

        Returns:
            Dictionary storing old parameters for potential reversion
        """
        old_params = {}

        for name, param in self.model.named_parameters():
            old_params[name] = param.data.clone()
            noise = torch.randn_like(param.data) * proposal_std
            param.data.add_(noise)

        return old_params

    def revert_parameters(self, old_params: Dict[str, torch.Tensor]) -> None:
        """
        Revert model parameters to previous state.

        Args:
            old_params: Dictionary of old parameter values
        """
        for name, param in self.model.named_parameters():
            param.data.copy_(old_params[name])

    def metropolis_hastings_step(self, data_loader: DataLoader,
                                  proposal_std: float = 0.01) -> Tuple[bool, Dict[str, float]]:
        """
        Perform one Metropolis-Hastings step.

        Args:
            data_loader: DataLoader containing observations
            proposal_std: Standard deviation of proposal distribution

        Returns:
            Tuple of (accepted, metrics_dict)
        """
        # Compute current log posterior
        current_log_post, current_log_lik, current_log_prior = self.log_posterior(data_loader)

        # Propose new parameters
        old_params = self.propose_parameters(proposal_std)

        # Compute proposed log posterior
        proposed_log_post, proposed_log_lik, proposed_log_prior = self.log_posterior(data_loader)

        # Metropolis-Hastings acceptance ratio with temperature
        log_acceptance_ratio = (proposed_log_post - current_log_post) / self.temperature

        # Accept or reject
        accept_threshold = torch.log(torch.rand(1).to(self.local_rank)).item()
        accepted = accept_threshold < log_acceptance_ratio.item()

        if accepted:
            # Accept proposal
            self.acceptance_count += 1
            final_log_post = proposed_log_post.item()
            final_log_lik = proposed_log_lik.item()
            final_log_prior = proposed_log_prior.item()
        else:
            # Reject proposal - revert to old parameters
            self.revert_parameters(old_params)
            final_log_post = current_log_post.item()
            final_log_lik = current_log_lik.item()
            final_log_prior = current_log_prior.item()

        self.proposal_count += 1

        # Prepare metrics for logging
        metrics = {
            'log_posterior': final_log_post,
            'log_likelihood': final_log_lik,
            'log_prior': final_log_prior,
            'log_acceptance_ratio': log_acceptance_ratio.item(),
            'accepted': int(accepted)
        }

        return accepted, metrics

    def get_acceptance_rate(self) -> float:
        """
        Get acceptance rate for Metropolis-Hastings proposals.

        Returns:
            Acceptance rate as a fraction
        """
        if self.proposal_count == 0:
            return 0.0
        return self.acceptance_count / self.proposal_count

    def get_swap_acceptance_rate(self) -> float:
        """
        Get acceptance rate for replica swaps.

        Returns:
            Swap acceptance rate as a fraction
        """
        if self.swap_proposal_count == 0:
            return 0.0
        return self.swap_acceptance_count / self.swap_proposal_count

    def save_sample(self) -> None:
        """
        Save current parameter configuration as a sample.
        """
        sample = {}
        for name, param in self.model.named_parameters():
            sample[name] = param.data.cpu().clone()
        self.samples.append(sample)

    def compute_param_statistics(self) -> Dict[str, float]:
        """
        Compute statistics of current parameters.

        Returns:
            Dictionary of parameter statistics
        """
        all_params = []
        param_norms = {}

        for name, param in self.model.named_parameters():
            flat_param = param.data.flatten()
            all_params.append(flat_param)
            param_norms[f'param_norm/{name}'] = torch.norm(param.data).item()

        all_params = torch.cat(all_params)

        stats = {
            'param_mean': all_params.mean().item(),
            'param_std': all_params.std().item(),
            'param_min': all_params.min().item(),
            'param_max': all_params.max().item(),
            **param_norms
        }

        return stats

    def get_samples_as_tensor(self, param_name: str) -> torch.Tensor:
        """
        Get all samples for a specific parameter as a tensor.

        Args:
            param_name: Name of the parameter

        Returns:
            Tensor of shape (num_samples, *param_shape)
        """
        return torch.stack([s[param_name] for s in self.samples])


def replica_exchange_swap(model: nn.Module, temperature: float,
                          log_posterior: float, local_rank: int,
                          num_gpus: int) -> Tuple[bool, float]:
    """
    Attempt to swap configurations between adjacent temperature replicas.

    The swap acceptance probability is:
    α = min(1, exp[(1/T_i - 1/T_j)(log p(θ_j|D) - log p(θ_i|D))])

    This function implements a symmetric exchange protocol where both replicas
    participate in the swap decision.

    Args:
        model: Neural network model
        temperature: Current temperature
        log_posterior: Current log posterior
        local_rank: GPU rank
        num_gpus: Total number of GPUs

    Returns:
        Tuple of (swap_occurred, new_log_posterior)
    """
    if num_gpus < 2:
        return False, log_posterior

    # Determine swap partner (adjacent replica)
    # For 2 GPUs: rank 0 swaps with rank 1, rank 1 swaps with rank 0
    if local_rank == 0:
        partner_rank = 1
    elif local_rank == num_gpus - 1:
        partner_rank = num_gpus - 2
    else:
        # For >2 GPUs, alternate between swapping with left or right neighbor
        # to avoid conflicts (even ranks swap right, odd ranks swap left)
        if local_rank % 2 == 0:
            partner_rank = local_rank + 1
        else:
            partner_rank = local_rank - 1

    swap_occurred = False
    new_log_posterior = log_posterior

    # Exchange temperature and log posterior information
    partner_temp = torch.tensor([0.0]).to(local_rank)
    partner_log_post = torch.tensor([0.0]).to(local_rank)

    my_temp = torch.tensor([temperature]).to(local_rank)
    my_log_post = torch.tensor([log_posterior]).to(local_rank)

    # Synchronize - both replicas send and receive
    if local_rank < partner_rank:
        # Lower rank sends first, then receives
        dist.send(my_temp, dst=partner_rank)
        dist.send(my_log_post, dst=partner_rank)
        dist.recv(partner_temp, src=partner_rank)
        dist.recv(partner_log_post, src=partner_rank)
    else:
        # Higher rank receives first, then sends
        dist.recv(partner_temp, src=partner_rank)
        dist.recv(partner_log_post, src=partner_rank)
        dist.send(my_temp, dst=partner_rank)
        dist.send(my_log_post, dst=partner_rank)

    # Compute swap acceptance probability
    # α = min(1, exp[(β_i - β_j)(E_j - E_i)])
    # where β = 1/T and E = -log p(θ|D)
    beta_i = 1.0 / temperature
    beta_j = 1.0 / partner_temp.item()
    energy_i = -log_posterior  # Energy = -log posterior
    energy_j = -partner_log_post.item()

    log_acceptance = (beta_i - beta_j) * (energy_j - energy_i)

    # Both replicas make the same decision using synchronized random number
    # Only the lower-rank replica generates the random number
    if local_rank < partner_rank:
        random_val = torch.rand(1).to(local_rank)
        dist.send(random_val, dst=partner_rank)
    else:
        random_val = torch.zeros(1).to(local_rank)
        dist.recv(random_val, src=partner_rank)

    # Decide whether to accept swap
    accept_swap = torch.log(random_val).item() < log_acceptance

    if accept_swap:
        # Exchange parameters with partner
        for param in model.parameters():
            # Create buffer for partner's parameters
            partner_param = torch.zeros_like(param.data)

            if local_rank < partner_rank:
                dist.send(param.data, dst=partner_rank)
                dist.recv(partner_param, src=partner_rank)
            else:
                dist.recv(partner_param, src=partner_rank)
                dist.send(param.data, dst=partner_rank)

            param.data.copy_(partner_param)

        swap_occurred = True
        new_log_posterior = partner_log_post.item()

        logger.info(
            f"Replica {local_rank} (T={temperature:.3f}): "
            f"Swap ACCEPTED with replica {partner_rank} (T={partner_temp.item():.3f}), "
            f"log_acceptance={log_acceptance:.3f}"
        )
    else:
        logger.info(
            f"Replica {local_rank} (T={temperature:.3f}): "
            f"Swap REJECTED with replica {partner_rank} (T={partner_temp.item():.3f}), "
            f"log_acceptance={log_acceptance:.3f}"
        )

    return swap_occurred, new_log_posterior


def run_parallel_tempering(model: nn.Module, data_loader: DataLoader,
                           sampler: ParallelTemperingMCMC,
                           num_iterations: int = 1000,
                           burn_in: int = 200,
                           thinning: int = 5,
                           swap_interval: int = 10,
                           proposal_std: float = 0.01,
                           local_rank: int = 0,
                           num_gpus: int = 2,
                           use_wandb: bool = True) -> None:
    """
    Run parallel tempering MCMC sampling with W&B logging.

    Args:
        model: Neural network model
        data_loader: DataLoader containing observations
        sampler: ParallelTemperingMCMC instance
        num_iterations: Total number of MCMC iterations
        burn_in: Number of burn-in iterations to discard
        thinning: Keep every nth sample after burn-in
        swap_interval: Attempt replica swaps every n iterations
        proposal_std: Standard deviation for parameter proposals
        local_rank: GPU rank
        num_gpus: Total number of GPUs
        use_wandb: Whether to log to W&B
    """
    logger.info(f"Replica {local_rank}: Starting parallel tempering MCMC")
    logger.info(f"Replica {local_rank}: Temperature = {sampler.temperature:.3f}")

    current_log_posterior, current_log_lik, current_log_prior = sampler.log_posterior(data_loader)
    current_log_posterior = current_log_posterior.item()

    for iteration in range(num_iterations):
        # Metropolis-Hastings step
        accepted, mh_metrics = sampler.metropolis_hastings_step(data_loader, proposal_std)

        if accepted:
            current_log_posterior = mh_metrics['log_posterior']
            current_log_lik = mh_metrics['log_likelihood']
            current_log_prior = mh_metrics['log_prior']

        # Compute parameter statistics
        param_stats = sampler.compute_param_statistics()

        # Log metrics to W&B (only from rank 0)
        if use_wandb and local_rank == 0:
            log_dict = {
                f'replica_{local_rank}/log_posterior': current_log_posterior,
                f'replica_{local_rank}/log_likelihood': current_log_lik,
                f'replica_{local_rank}/log_prior': current_log_prior,
                f'replica_{local_rank}/acceptance_rate': sampler.get_acceptance_rate(),
                f'replica_{local_rank}/log_acceptance_ratio': mh_metrics['log_acceptance_ratio'],
                f'replica_{local_rank}/accepted': mh_metrics['accepted'],
                f'replica_{local_rank}/temperature': sampler.temperature,
                f'replica_{local_rank}/iteration': iteration,
            }

            # Add parameter statistics
            for key, value in param_stats.items():
                log_dict[f'replica_{local_rank}/{key}'] = value

            wandb.log(log_dict, step=iteration)

        # Attempt replica exchange swaps periodically
        swap_occurred = False
        if iteration % swap_interval == 0 and iteration > 0:
            swap_occurred, new_log_posterior = replica_exchange_swap(
                model, sampler.temperature, current_log_posterior,
                local_rank, num_gpus
            )

            if swap_occurred:
                sampler.swap_acceptance_count += 1
                current_log_posterior = new_log_posterior

                # Recompute likelihood and prior after swap
                _, current_log_lik, current_log_prior = sampler.log_posterior(data_loader)
                current_log_lik = current_log_lik.item()
                current_log_prior = current_log_prior.item()

            sampler.swap_proposal_count += 1

            # Log swap metrics (only from rank 0)
            if use_wandb and local_rank == 0:
                wandb.log({
                    f'replica_{local_rank}/swap_occurred': int(swap_occurred),
                    f'replica_{local_rank}/swap_acceptance_rate': sampler.get_swap_acceptance_rate(),
                }, step=iteration)

        # Save samples after burn-in with thinning
        if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
            sampler.save_sample()
            sampler.log_posteriors.append(current_log_posterior)
            sampler.log_likelihoods.append(current_log_lik)
            sampler.log_priors.append(current_log_prior)

            # Log sample collection for coldest chain
            if use_wandb and local_rank == 0:
                wandb.log({
                    'samples_collected': len(sampler.samples),
                }, step=iteration)

        # Periodic detailed logging
        if iteration % 100 == 0:
            logger.info(
                f"Replica {local_rank} - Iter {iteration}/{num_iterations}: "
                f"Log Post = {current_log_posterior:.2f}, "
                f"Accept Rate = {sampler.get_acceptance_rate():.3f}, "
                f"Swap Rate = {sampler.get_swap_acceptance_rate():.3f}"
            )

            # Log histogram of parameters to W&B (only for rank 0 to avoid clutter)
            if use_wandb and local_rank == 0 and iteration % 500 == 0:
                for name, param in model.named_parameters():
                    wandb.log({
                        f'param_hist/{name}': wandb.Histogram(param.data.cpu().numpy().flatten())
                    }, step=iteration)

    logger.info(f"Replica {local_rank}: Sampling completed")
    logger.info(f"Replica {local_rank}: Collected {len(sampler.samples)} samples")
    logger.info(f"Replica {local_rank}: Final acceptance rate = {sampler.get_acceptance_rate():.3f}")
    logger.info(f"Replica {local_rank}: Final swap rate = {sampler.get_swap_acceptance_rate():.3f}")

    # Final summary metrics (only from rank 0)
    if use_wandb and local_rank == 0:
        wandb.log({
            f'replica_{local_rank}/final_acceptance_rate': sampler.get_acceptance_rate(),
            f'replica_{local_rank}/final_swap_rate': sampler.get_swap_acceptance_rate(),
            f'replica_{local_rank}/total_samples': len(sampler.samples),
        })


def create_posterior_visualization(sampler: ParallelTemperingMCMC,
                                   local_rank: int) -> Optional[wandb.Image]:
    """
    Create visualization of posterior samples for W&B.

    Args:
        sampler: ParallelTemperingMCMC instance
        local_rank: GPU rank

    Returns:
        W&B Image object or None
    """
    if len(sampler.samples) == 0:
        return None

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Log posterior trace
        axes[0, 0].plot(sampler.log_posteriors, alpha=0.7)
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Log Posterior')
        axes[0, 0].set_title(f'Log Posterior Trace (T={sampler.temperature:.3f})')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: First parameter trace
        param_name = 'fc1.weight'
        param_values = [s[param_name].flatten()[0].item() for s in sampler.samples]
        axes[0, 1].plot(param_values, alpha=0.7)
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel(f'{param_name}[0]')
        axes[0, 1].set_title('Parameter Trace')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Parameter distribution
        axes[1, 0].hist(param_values, bins=50, density=True, alpha=0.7)
        axes[1, 0].set_xlabel(f'{param_name}[0]')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Parameter Posterior Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Log likelihood vs log prior
        axes[1, 1].scatter(sampler.log_priors, sampler.log_likelihoods,
                          alpha=0.5, s=10)
        axes[1, 1].set_xlabel('Log Prior')
        axes[1, 1].set_ylabel('Log Likelihood')
        axes[1, 1].set_title('Log Likelihood vs Log Prior')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to W&B image
        wandb_image = wandb.Image(fig)
        plt.close(fig)

        return wandb_image
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        return None


def save_samples(sampler: ParallelTemperingMCMC, save_dir: str,
                local_rank: int) -> None:
    """
    Save MCMC samples to disk.

    Args:
        sampler: ParallelTemperingMCMC instance
        save_dir: Directory to save samples
        local_rank: GPU rank
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save samples
    samples_file = os.path.join(save_dir, f"samples_replica_{local_rank}.pt")
    torch.save({
        'samples': sampler.samples,
        'log_posteriors': sampler.log_posteriors,
        'log_likelihoods': sampler.log_likelihoods,
        'log_priors': sampler.log_priors,
        'temperature': sampler.temperature,
        'acceptance_rate': sampler.get_acceptance_rate(),
        'swap_acceptance_rate': sampler.get_swap_acceptance_rate()
    }, samples_file)

    logger.info(f"Replica {local_rank}: Samples saved to {samples_file}")


def main() -> None:
    """
    Main function for parallel tempering MCMC sampling.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Parallel Tempering MCMC with DeepSpeed and W&B"
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank passed from distributed launcher")
    parser.add_argument("--num_iterations", type=int, default=1000,
                       help="Number of MCMC iterations")
    parser.add_argument("--burn_in", type=int, default=200,
                       help="Number of burn-in iterations")
    parser.add_argument("--thinning", type=int, default=5,
                       help="Thinning interval for samples")
    parser.add_argument("--swap_interval", type=int, default=10,
                       help="Interval for replica exchange attempts")
    parser.add_argument("--proposal_std", type=float, default=0.01,
                       help="Standard deviation for parameter proposals")
    parser.add_argument("--prior_std", type=float, default=1.0,
                       help="Standard deviation of prior distribution")
    parser.add_argument("--noise_std", type=float, default=0.1,
                       help="Standard deviation of observation noise")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for likelihood computation")
    parser.add_argument("--save_dir", type=str, default="./pt_samples",
                       help="Directory to save samples")
    parser.add_argument("--wandb_project", type=str,
                       default="parallel-tempering-mcmc",
                       help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="W&B entity (username or team name)")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Name for this experiment run")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable W&B logging")

    # DeepSpeed config will be created from embedded config
    args = parser.parse_args()

    # Create temporary DeepSpeed config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(DEEPSPEED_CONFIG, f, indent=2)
        deepspeed_config_path = f.name

    logger.info(f"Created temporary DeepSpeed config at: {deepspeed_config_path}")

    # Add deepspeed_config to args
    args.deepspeed_config = deepspeed_config_path

    # Initialize distributed training
    deepspeed.init_distributed()
    local_rank = args.local_rank
    num_gpus = torch.cuda.device_count()

    # Set device
    torch.cuda.set_device(local_rank)

    # Suppress NCCL warning
    os.environ['TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING'] = '0'

    # Initialize W&B (only on rank 0)
    use_wandb = not args.no_wandb
    if use_wandb and local_rank == 0:
        # Initialize W&B on rank 0
        run_name = args.experiment_name or f"pt_mcmc_{num_gpus}gpus"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                'num_gpus': num_gpus,
                'num_iterations': args.num_iterations,
                'burn_in': args.burn_in,
                'thinning': args.thinning,
                'swap_interval': args.swap_interval,
                'proposal_std': args.proposal_std,
                'prior_std': args.prior_std,
                'noise_std': args.noise_std,
                'batch_size': args.batch_size,
                'T_min': 1.0,
                'T_max': 10.0,
            }
        )
        logger.info(f"W&B initialized: {wandb.run.url}")

    # Wait for rank 0 to initialize W&B
    if dist.is_initialized():
        dist.barrier()

    # Create temperature schedule (geometric spacing)
    # T_i = T_min * (T_max / T_min)^(i / (num_gpus - 1))
    T_min = 1.0  # Coldest chain samples from true posterior
    T_max = 10.0  # Hottest chain explores more freely
    temperatures = [T_min * (T_max / T_min) ** (i / max(num_gpus - 1, 1))
                   for i in range(num_gpus)]

    logger.info(f"Temperature schedule: {temperatures}")

    # Log temperature schedule to W&B
    if use_wandb and local_rank == 0:
        wandb.config.update({'temperatures': temperatures})

    # Create dataset and dataloader
    dataset = SyntheticDataset(num_samples=1000, input_dim=10,
                               noise_std=args.noise_std)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Initialize model
    model = BayesianMLP(input_dim=10, hidden_dim=50, output_dim=1)
    model = model.to(local_rank)

    # Note: We don't use DeepSpeed's optimizer here since we're doing MCMC
    # but we still initialize with DeepSpeed for distributed communication
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # Initialize parallel tempering sampler
    sampler = ParallelTemperingMCMC(
        model=model_engine.module,
        temperatures=temperatures,
        prior_std=args.prior_std,
        noise_std=args.noise_std,
        local_rank=local_rank
    )

    # Run parallel tempering
    run_parallel_tempering(
        model=model_engine.module,
        data_loader=data_loader,
        sampler=sampler,
        num_iterations=args.num_iterations,
        burn_in=args.burn_in,
        thinning=args.thinning,
        swap_interval=args.swap_interval,
        proposal_std=args.proposal_std,
        local_rank=local_rank,
        num_gpus=num_gpus,
        use_wandb=use_wandb
    )

    # Save samples
    save_samples(sampler, args.save_dir, local_rank)

    # Create and log final visualizations to W&B (only on rank 0)
    if use_wandb and local_rank == 0:
        logger.info("Creating final visualizations...")

        # Create visualization for coldest chain
        posterior_viz = create_posterior_visualization(sampler, local_rank)
        if posterior_viz:
            wandb.log({
                'final_posterior_visualization': posterior_viz
            })

        # Log final summary statistics
        wandb.run.summary['final_acceptance_rate'] = sampler.get_acceptance_rate()
        wandb.run.summary['final_swap_rate'] = sampler.get_swap_acceptance_rate()
        wandb.run.summary['total_samples_collected'] = len(sampler.samples)

        # Create summary table
        summary_data = []
        for rank in range(num_gpus):
            summary_data.append([
                rank,
                temperatures[rank],
                sampler.get_acceptance_rate() if rank == local_rank else None,
                sampler.get_swap_acceptance_rate() if rank == local_rank else None
            ])

        summary_table = wandb.Table(
            columns=['Replica', 'Temperature', 'Acceptance Rate', 'Swap Rate'],
            data=summary_data
        )
        wandb.log({'replica_summary': summary_table})

        wandb.finish()

    # Clean up temporary config file
    try:
        os.unlink(deepspeed_config_path)
        logger.info("Cleaned up temporary DeepSpeed config")
    except Exception as e:
        logger.warning(f"Failed to clean up temp config: {e}")

    logger.info("Parallel tempering MCMC completed successfully!")


if __name__ == "__main__":
    main()

---
sidebar_position: 1
---

# Bayesian Neural Networks

Train Bayesian neural networks using parallel tempering MCMC for uncertainty estimation.

## Overview

This example demonstrates:
- Parallel tempering (replica exchange) MCMC
- Multi-GPU distributed Bayesian inference
- Temperature-based chain swaps
- Multimodal posterior exploration

**Task:** Bayesian inference for neural network weights

---

## The Bayesian Framework

### From Conditional Probability to Bayes' Theorem

The Bayesian framework emerges from a simple question: **How should we update our beliefs when we observe new evidence?**

Starting with the definition of conditional probability:

$$
P(A | B) = \frac{P(A \cap B)}{P(B)}
$$

We can write the joint probability two ways:

$$
P(A \cap B) = P(A | B) \cdot P(B) = P(B | A) \cdot P(A)
$$

Rearranging gives us **Bayes' Theorem**:

$$
P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

### Bayes' Theorem for Inference

In the context of statistical inference, we replace $A$ with parameters $\theta$ and $B$ with observed data $\mathcal{D}$:

$$
\underbrace{P(\theta | \mathcal{D})}_{\text{Posterior}} = \frac{\overbrace{P(\mathcal{D} | \theta)}^{\text{Likelihood}} \cdot \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(\mathcal{D})}_{\text{Evidence}}}
$$

Or more compactly:

$$
\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
$$

### Why Do Statisticians Use This Framework?

The Bayesian approach provides several fundamental advantages:

```mermaid
graph TB
    subgraph "Bayesian Reasoning"
        PRIOR["Prior P(θ)<br/>What we believe before seeing data"]
        DATA["Data D<br/>Observed evidence"]
        LIKELIHOOD["Likelihood P(D|θ)<br/>How probable is data given parameters?"]
        POSTERIOR["Posterior P(θ|D)<br/>Updated beliefs after seeing data"]

        PRIOR --> COMBINE["Bayes' Theorem"]
        DATA --> LIKELIHOOD
        LIKELIHOOD --> COMBINE
        COMBINE --> POSTERIOR
    end

    style PRIOR fill:#0f2f4d,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
    style POSTERIOR fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    style LIKELIHOOD fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff
```

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Parameters | Fixed but unknown | Random variables with distributions |
| Uncertainty | Confidence intervals (long-run frequency) | Credible intervals (probability statements) |
| Prior knowledge | Not formally incorporated | Explicitly encoded in prior |
| Results | Point estimates + p-values | Full posterior distribution |
| Interpretation | "If we repeated this experiment..." | "Given this data, the probability is..." |

### The Challenge: Computing the Posterior

The evidence (marginal likelihood) requires integrating over all possible parameter values:

$$
P(\mathcal{D}) = \int P(\mathcal{D} | \theta) \cdot P(\theta) \, d\theta
$$

For neural networks with millions of parameters, this integral is **intractable**. We cannot compute it analytically.

**This is why we need MCMC.**

---

## Markov Chain Monte Carlo (MCMC)

### The Core Idea

MCMC is a clever solution to an impossible problem: **instead of computing the posterior analytically, we generate samples from it.**

```mermaid
graph LR
    subgraph "MCMC Sampling"
        START["Start at θ₀"] --> S1["Sample θ₁"]
        S1 --> S2["Sample θ₂"]
        S2 --> S3["Sample θ₃"]
        S3 --> DOTS["..."]
        DOTS --> SN["Sample θₙ"]
    end

    SN --> APPROX["Approximate posterior<br/>with empirical distribution"]
```

**Key Insight:** If we construct a Markov chain whose **stationary distribution** is the posterior $P(\theta | \mathcal{D})$, then after enough steps, the samples will be distributed according to the posterior.

### Why "Monte Carlo"?

Monte Carlo methods use random sampling to solve deterministic problems. Instead of computing:

$$
\mathbb{E}_{P(\theta|\mathcal{D})}[f(\theta)] = \int f(\theta) \cdot P(\theta | \mathcal{D}) \, d\theta
$$

We approximate with samples:

$$
\mathbb{E}_{P(\theta|\mathcal{D})}[f(\theta)] \approx \frac{1}{N} \sum_{i=1}^{N} f(\theta^{(i)}), \quad \theta^{(i)} \sim P(\theta | \mathcal{D})
$$

### Why "Markov Chain"?

A Markov chain has the property that the next state depends only on the current state:

$$
P(\theta^{(t+1)} | \theta^{(t)}, \theta^{(t-1)}, \ldots, \theta^{(0)}) = P(\theta^{(t+1)} | \theta^{(t)})
$$

This memoryless property makes the chain computationally tractable while still able to explore the full parameter space.

### The Metropolis-Hastings Algorithm

The most fundamental MCMC algorithm:

**Algorithm:**
1. Start at some initial $\theta^{(0)}$
2. For $t = 1, 2, \ldots, N$:
   - Propose: $\theta^* \sim Q(\theta^* | \theta^{(t-1)})$
   - Compute acceptance ratio:
   $$
   \alpha = \min\left(1, \frac{P(\theta^* | \mathcal{D}) \cdot Q(\theta^{(t-1)} | \theta^*)}{P(\theta^{(t-1)} | \mathcal{D}) \cdot Q(\theta^* | \theta^{(t-1)})}\right)
   $$
   - Accept with probability $\alpha$:
   $$
   \theta^{(t)} = \begin{cases} \theta^* & \text{with probability } \alpha \\ \theta^{(t-1)} & \text{with probability } 1 - \alpha \end{cases}
   $$

For symmetric proposals $Q(\theta^* | \theta) = Q(\theta | \theta^*)$, this simplifies to:

$$
\alpha = \min\left(1, \frac{P(\theta^* | \mathcal{D})}{P(\theta^{(t-1)} | \mathcal{D})}\right)
$$

**Key Property:** We only need the ratio of posteriors, so the intractable normalizing constant $P(\mathcal{D})$ cancels out!

$$
\frac{P(\theta^* | \mathcal{D})}{P(\theta^{(t-1)} | \mathcal{D})} = \frac{P(\mathcal{D} | \theta^*) \cdot P(\theta^*)}{P(\mathcal{D} | \theta^{(t-1)}) \cdot P(\theta^{(t-1)})}
$$

### Visualizing MCMC

```mermaid
graph TB
    subgraph "Metropolis-Hastings Step"
        CURRENT["Current θ(t)"]
        PROPOSE["Propose θ*"]
        COMPUTE["Compute α = P(θ*|D) / P(θ(t)|D)"]
        ACCEPT["Accept: θ(t+1) = θ*"]
        REJECT["Reject: θ(t+1) = θ(t)"]

        CURRENT --> PROPOSE
        PROPOSE --> COMPUTE
        COMPUTE -->|"U < α"| ACCEPT
        COMPUTE -->|"U ≥ α"| REJECT
    end

    style ACCEPT fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    style REJECT fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
```

---

## The Problem with Standard MCMC

### Multimodal Posteriors

Neural network posteriors are notoriously **multimodal** — they have many peaks separated by valleys of low probability:

```mermaid
graph TB
    subgraph "Multimodal Posterior Landscape"
        M1["Mode 1<br/>(local optimum)"]
        M2["Mode 2<br/>(global optimum)"]
        M3["Mode 3<br/>(local optimum)"]

        VALLEY1["Low probability<br/>valley"]
        VALLEY2["Low probability<br/>valley"]

        M1 --- VALLEY1
        VALLEY1 --- M2
        M2 --- VALLEY2
        VALLEY2 --- M3
    end
```

**The Problem:** Standard MCMC chains get **trapped** in one mode. They cannot cross the low-probability valleys to discover other modes.

### Why Does This Matter?

If we only sample from one mode:
- Our uncertainty estimates are **overconfident**
- We miss important alternative parameter configurations
- Predictions may be **biased** toward one solution

---

## Bayesian Neural Networks: Purpose and Benefits

### What Makes Neural Networks "Bayesian"?

In standard neural networks, we find a **single point estimate** of the weights $\hat{\mathbf{w}}$ by minimizing a loss function.

In Bayesian neural networks, we treat weights as **random variables** and compute the **full posterior distribution** $P(\mathbf{w} | \mathcal{D})$.

```mermaid
graph LR
    subgraph "Standard NN"
        INPUT1["Input x"] --> NN1["Network with<br/>fixed weights ŵ"]
        NN1 --> OUTPUT1["Single prediction<br/>ŷ = f(x; ŵ)"]
    end

    subgraph "Bayesian NN"
        INPUT2["Input x"] --> NN2["Network with<br/>weight distribution P(w|D)"]
        NN2 --> OUTPUT2["Predictive distribution<br/>P(y|x,D)"]
    end

    style OUTPUT1 fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    style OUTPUT2 fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
```

### The Bayesian Predictive Distribution

Instead of a point prediction, we integrate over all possible weights:

$$
P(y | x, \mathcal{D}) = \int P(y | x, \mathbf{w}) \cdot P(\mathbf{w} | \mathcal{D}) \, d\mathbf{w}
$$

In practice, we approximate with MCMC samples:

$$
P(y | x, \mathcal{D}) \approx \frac{1}{N} \sum_{i=1}^{N} P(y | x, \mathbf{w}^{(i)}), \quad \mathbf{w}^{(i)} \sim P(\mathbf{w} | \mathcal{D})
$$

### Why Use Bayesian Neural Networks?

| Capability | How It Works |
|------------|--------------|
| **Uncertainty Quantification** | The spread of the predictive distribution tells us how confident the model is |
| **Robust Predictions** | Averaging over many weight configurations reduces overfitting |
| **Out-of-Distribution Detection** | High uncertainty on unfamiliar inputs |
| **Principled Regularization** | Priors act as regularizers (e.g., weight decay ≈ Gaussian prior) |
| **Model Comparison** | Marginal likelihood enables formal model selection |

### Types of Uncertainty

Bayesian NNs distinguish two types of uncertainty:

$$
\underbrace{\text{Var}[y|x,\mathcal{D}]}_{\text{Total Uncertainty}} = \underbrace{\mathbb{E}[\text{Var}[y|x,\mathbf{w}]]}_{\text{Aleatoric (data noise)}} + \underbrace{\text{Var}[\mathbb{E}[y|x,\mathbf{w}]]}_{\text{Epistemic (model uncertainty)}}
$$

- **Aleatoric uncertainty**: Inherent noise in the data (irreducible)
- **Epistemic uncertainty**: Uncertainty due to limited data (reducible with more data)

```mermaid
graph TB
    subgraph "Uncertainty Decomposition"
        TOTAL["Total Predictive Uncertainty"]
        ALEA["Aleatoric<br/>(Data noise)"]
        EPIS["Epistemic<br/>(Model uncertainty)"]

        TOTAL --> ALEA
        TOTAL --> EPIS

        ALEA --> ALEA_EX["Cannot reduce<br/>with more data"]
        EPIS --> EPIS_EX["Reduces with<br/>more data"]
    end

    style ALEA fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    style EPIS fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
```

---

## Parallel Tempering: The Solution

### The Temperature Concept

Parallel tempering introduces a **temperature** parameter $T$ that modifies the posterior:

$$
P_T(\theta | \mathcal{D}) \propto P(\mathcal{D} | \theta)^{1/T} \cdot P(\theta)
$$

Or equivalently, in log space:

$$
\log P_T(\theta | \mathcal{D}) = \frac{1}{T} \log P(\mathcal{D} | \theta) + \log P(\theta) + \text{const}
$$

### What Temperature Does

| Temperature | Effect on Posterior | Behavior |
|-------------|---------------------|----------|
| $T = 1$ | Original posterior | Samples from true target |
| $T > 1$ | Flattened posterior | Easier to cross barriers |
| $T \to \infty$ | Approaches prior | Random walk exploration |

```mermaid
graph LR
    subgraph "Effect of Temperature"
        COLD["T = 1 (Cold)<br/>Sharp peaks<br/>Accurate but trapped"]
        WARM["T = 2 (Warm)<br/>Softer peaks<br/>Better exploration"]
        HOT["T = 4 (Hot)<br/>Nearly flat<br/>Free exploration"]
    end

    style COLD fill:#1b4a75,stroke:#5590c0,stroke-width:1.5px,color:#ffffff
    style WARM fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff
    style HOT fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
```

### Visualizing Temperature Effects

At temperature $T$, the posterior becomes:

$$
P_T(\theta) \propto P(\theta)^{1/T}
$$

For $T > 1$:
- Peaks become **shorter** (less concentrated)
- Valleys become **shallower** (easier to cross)
- The landscape becomes **smoother**

**Mathematical Intuition:** If the original posterior has a barrier with probability ratio $10^{-6}$, at $T=2$ this becomes $(10^{-6})^{1/2} = 10^{-3}$, making it 1000× easier to cross!

---

## The Replica Exchange Algorithm

### Why Exchange Replicas?

Running hot chains alone isn't useful — they don't sample from the correct distribution. The key insight is:

1. **Hot chains** explore freely and find new modes
2. **Cold chains** sample accurately from discovered modes
3. **Swaps** transfer discoveries from hot chains to cold chains

```mermaid
graph TB
    subgraph "Replica Exchange Process"
        subgraph "Cold Chain T=1"
            C1["Accurate sampling<br/>in current mode"]
        end

        subgraph "Hot Chain T=4"
            H1["Discovers<br/>new mode"]
        end

        C1 <-->|"Swap states"| H1

        subgraph "After Swap"
            C2["Cold chain now<br/>samples new mode"]
            H2["Hot chain continues<br/>exploring"]
        end

        C1 -.-> C2
        H1 -.-> H2
    end
```

### The Swap Acceptance Criterion

For chains $i$ and $j$ at temperatures $T_i$ and $T_j$, the swap acceptance probability is:

$$
\alpha_{swap} = \min\left(1, \exp(\Delta)\right)
$$

Where:

$$
\Delta = \left(\frac{1}{T_i} - \frac{1}{T_j}\right) \cdot \left(\log P(\mathcal{D} | \theta_j) - \log P(\mathcal{D} | \theta_i)\right)
$$

**Why This Formula?**

To maintain detailed balance (ensuring the combined system has the correct stationary distribution), we need:

$$
P_i(\theta_i) P_j(\theta_j) \cdot \alpha(\theta_i \leftrightarrow \theta_j) = P_i(\theta_j) P_j(\theta_i) \cdot \alpha(\theta_j \leftrightarrow \theta_i)
$$

This leads to the Metropolis criterion for swaps:

$$
\alpha = \min\left(1, \frac{P_i(\theta_j) P_j(\theta_i)}{P_i(\theta_i) P_j(\theta_j)}\right)
$$

Substituting $P_T(\theta) \propto P(\mathcal{D}|\theta)^{1/T} P(\theta)$:

$$
\alpha = \min\left(1, \frac{P(\mathcal{D}|\theta_j)^{1/T_i} P(\mathcal{D}|\theta_i)^{1/T_j}}{P(\mathcal{D}|\theta_i)^{1/T_i} P(\mathcal{D}|\theta_j)^{1/T_j}}\right)
$$

Taking logs gives our formula for $\Delta$.

### When Are Swaps Accepted?

Consider chains at $T_i = 1$ (cold) and $T_j = 2$ (warm):

$$
\Delta = \left(1 - 0.5\right) \cdot \left(\log P_j - \log P_i\right) = 0.5 \cdot (\log P_j - \log P_i)
$$

- If the **hot chain** found a **better** state ($\log P_j > \log P_i$): $\Delta > 0$, swap likely accepted
- If the **cold chain** has a **better** state ($\log P_i > \log P_j$): $\Delta < 0$, swap less likely

**This is exactly what we want:** good discoveries propagate from hot to cold chains!

### The Temperature Ladder

Choosing temperatures is crucial. A geometric spacing works well:

$$
T_k = T_{min} \cdot \left(\frac{T_{max}}{T_{min}}\right)^{k/(K-1)}, \quad k = 0, 1, \ldots, K-1
$$

For $K = 4$ GPUs with $T_{min} = 1$ and $T_{max} = 8$:

| GPU | $k$ | Temperature |
|-----|-----|-------------|
| 0 | 0 | $1.0$ |
| 1 | 1 | $2.0$ |
| 2 | 2 | $4.0$ |
| 3 | 3 | $8.0$ |

**Why Geometric Spacing?**

Work in inverse temperature $\beta = 1/T$. For a swap between adjacent chains, a second-order expansion of the acceptance rate gives

$$
\mathbb{E}[\alpha_{\text{swap}}] \;\approx\; \text{function of } \Delta\beta \cdot \varsigma_{\ell}(\beta)
$$

where $\varsigma_\ell(\beta)$ is the standard deviation of the log-likelihood under chain $\beta$. Acceptance is uniform across the ladder when $\Delta\beta \cdot \varsigma_\ell$ is constant. Because $\varsigma_\ell$ typically scales roughly as $1/\beta$ — hotter chains explore a wider range of likelihoods — holding $\Delta\beta/\beta$ constant does the job, and a **constant ratio** between adjacent temperatures is exactly that. Geometric spacing is therefore an approximation that works well in practice, not an identity.

:::note Sizing the ladder
Kone & Kofke (2005) and Atchadé et al. (2011) analyse the optimal adjacent-swap acceptance rate and find $\approx 0.23$ under idealized assumptions — the same constant that appears in optimal-scaling results for random-walk Metropolis. In practice anything in **0.2–0.5** is healthy.

Diagnose the ladder by acceptance rate *per adjacent pair*, not on average:

- **A pair below ~0.1** is a bottleneck: the ladder is too sparse there and the chains are effectively disconnected. Insert an intermediate temperature.
- **A pair above ~0.8** is wasted compute: the two chains sample nearly the same distribution. Remove one.

The number of rungs needed grows roughly as $\sqrt{\text{model dimension}}$, which is why parallel tempering is expensive for large networks and why this example distributes rungs across GPUs.
:::

:::warning Two tempering conventions — check which one you are implementing
The $\Delta$ above tempers **only the likelihood**, leaving the prior at full strength:

$$p_\beta(\theta) \propto p(\mathcal{D}\mid\theta)^{\beta}\,p(\theta)$$

The alternative tempers the whole posterior, $p_\beta(\theta) \propto \left[p(\mathcal{D}\mid\theta)p(\theta)\right]^{\beta}$. The first is standard for Bayesian inference — it keeps the prior as a proper regularizer, so hot chains still cannot wander to $\|\theta\| \to \infty$; the second comes from the statistical-physics literature. **They give different swap formulas**: with a tempered prior the $\Delta$ term must include the log-prior difference. Mixing the two — computing $\Delta$ one way while the sampler targets the other — silently breaks detailed balance and the stationary distribution is not the posterior. The formula above is correct for the likelihood-only convention used in this example.
:::

### Checking that it worked

MCMC gives no convergence guarantee you can check directly; you can only look for evidence of failure. Two standard diagnostics, both of which parallel tempering makes cheap because you already have multiple chains:

**$\hat R$ (Gelman–Rubin), split-$\hat R$ variant.** Compares within-chain to between-chain variance for each scalar quantity of interest:

$$
\hat R = \sqrt{\frac{\widehat{\operatorname{Var}}^{+}(\psi)}{W}}, \qquad \widehat{\operatorname{Var}}^{+}(\psi) = \frac{n-1}{n}W + \frac{1}{n}B
$$

with $W$ the mean within-chain variance and $B$ the between-chain variance. $\hat R \to 1$ as chains mix. Vehtari et al. (2021) recommend **$\hat R < 1.01$**, tighter than the older 1.1 threshold.

**Effective sample size.** MCMC draws are autocorrelated, so $n$ samples carry less information than $n$ independent ones:

$$
n_{\text{eff}} = \frac{n}{1 + 2\sum_{k=1}^{\infty}\rho_k}
$$

with $\rho_k$ the lag-$k$ autocorrelation. Report $n_{\text{eff}}$, not $n$ — 100,000 draws at $n_{\text{eff}} = 50$ is 50 samples, and the Monte Carlo standard error is $\varsigma/\sqrt{n_{\text{eff}}}$.

:::danger These diagnostics are necessary, not sufficient — and weaker for BNNs
$\hat R \approx 1$ across chains that all became trapped in the *same* mode says nothing about the modes they all missed. For a neural network this is not a corner case: the posterior has enormous **exact symmetry**, since permuting hidden units and (for odd activations) flipping signs leaves the likelihood unchanged. A network with $H$ hidden units per layer has at least $H!\,2^{H}$ equivalent modes per layer.

Two consequences. Parameter-space $\hat R$ is close to meaningless — chains in permutation-equivalent modes look maximally disagreeing while representing the identical function. And it is why parallel tempering is being used here at all. **Compute diagnostics on function-space quantities** — predictions on held-out inputs, the log-likelihood — which are invariant to these symmetries.
:::

---

## Complete Parallel Tempering Algorithm

```mermaid
graph TB
    subgraph "Parallel Tempering MCMC"
        INIT["Initialize K chains<br/>at temperatures T₁, T₂, ..., Tₖ"]

        subgraph "Parallel MCMC Steps"
            MCMC1["Chain 1: MCMC step at T₁"]
            MCMC2["Chain 2: MCMC step at T₂"]
            MCMCK["Chain K: MCMC step at Tₖ"]
        end

        SWAP["Attempt swaps between<br/>adjacent chains"]

        COLLECT["Collect samples from<br/>cold chain (T=1)"]

        CHECK{"Enough<br/>samples?"}

        INIT --> MCMC1
        INIT --> MCMC2
        INIT --> MCMCK

        MCMC1 --> SWAP
        MCMC2 --> SWAP
        MCMCK --> SWAP

        SWAP --> COLLECT
        COLLECT --> CHECK

        CHECK -->|"No"| MCMC1
        CHECK -->|"No"| MCMC2
        CHECK -->|"No"| MCMCK
        CHECK -->|"Yes"| DONE["Return posterior samples"]
    end
```

### Algorithm Pseudocode

```
Algorithm: Parallel Tempering MCMC

Input: K temperatures T₁ < T₂ < ... < Tₖ, N iterations
Output: Samples from posterior P(θ|D)

1. Initialize chains θ₁, θ₂, ..., θₖ
2. For iteration t = 1 to N:

   # Parallel MCMC updates (one per GPU)
   3. For each chain k in parallel:
      - Propose θ* ~ Q(θ*|θₖ)
      - α = min(1, P(D|θ*)^(1/Tₖ) · P(θ*) / P(D|θₖ)^(1/Tₖ) · P(θₖ))
      - Accept θₖ ← θ* with probability α

   # Replica exchange (communication between GPUs)
   4. For k = 1 to K-1:
      - Compute Δ = (1/Tₖ - 1/Tₖ₊₁) · (log P(D|θₖ₊₁) - log P(D|θₖ))
      - If log(U) < Δ where U ~ Uniform(0,1):
        - Swap θₖ ↔ θₖ₊₁

   # Collect samples from cold chain
   5. If t > burn_in:
      - Store θ₁ as posterior sample

6. Return collected samples
```

---

## Quick Start

```bash
cd 04_bayesian_neuralnet

# SLURM submission (2 GPUs)
sbatch run_deepspeed.sh

# Direct execution
deepspeed --num_gpus=2 parallel_tempering_mcmc.py
```

## Model Architecture

```python
class BayesianMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

## Implementation Details

### 1. Temperature Assignment

Each GPU runs a chain at a different temperature:

```python
def get_temperature(rank, num_gpus, max_temp=4.0):
    """Assign temperature based on GPU rank."""
    if num_gpus == 1:
        return 1.0
    # Geometric spacing for uniform swap acceptance
    return max_temp ** (rank / (num_gpus - 1))

# Example with 4 GPUs:
# GPU 0: T=1.0 (cold - collect samples here)
# GPU 1: T=1.587
# GPU 2: T=2.52
# GPU 3: T=4.0 (hot - explore freely)
```

### 2. MCMC Sampling with Temperature

Each chain performs Metropolis-Hastings updates:

```python
def mcmc_step(model, data, temperature):
    """Single MCMC step with temperature scaling."""
    # Propose new parameters
    old_params = get_params(model)
    new_params = propose(old_params, step_size=0.01)

    # Compute tempered log posterior
    old_log_prob = log_likelihood(model, data) / temperature + log_prior(model)
    set_params(model, new_params)
    new_log_prob = log_likelihood(model, data) / temperature + log_prior(model)

    # Metropolis acceptance
    log_alpha = new_log_prob - old_log_prob
    if np.log(np.random.random()) < log_alpha:
        return True  # Accept
    else:
        set_params(model, old_params)
        return False  # Reject
```

### 3. Replica Exchange Between GPUs

```python
def attempt_swap(chain_i, chain_j, temp_i, temp_j):
    """Attempt swap between adjacent temperature chains."""
    # Compute log likelihoods (not tempered)
    log_lik_i = log_likelihood(chain_i, data)
    log_lik_j = log_likelihood(chain_j, data)

    # Swap acceptance criterion
    delta = (1/temp_i - 1/temp_j) * (log_lik_j - log_lik_i)

    if np.log(np.random.random()) < delta:
        # Swap parameters between chains
        params_i = get_params(chain_i)
        params_j = get_params(chain_j)
        set_params(chain_i, params_j)
        set_params(chain_j, params_i)
        return True
    return False
```

### 4. Log Posterior Computation

```python
def log_posterior(model, data, temperature=1.0):
    """Compute tempered log posterior."""
    x, y = data

    # Log likelihood (tempered)
    predictions = model(x)
    mse = F.mse_loss(predictions, y, reduction='sum')
    log_lik = -0.5 * mse / (noise_variance * temperature)

    # Log prior (not tempered - keeps regularization constant)
    log_prior = 0
    for param in model.parameters():
        log_prior -= 0.5 * prior_precision * (param ** 2).sum()

    return log_lik + log_prior
```

## DeepSpeed Configuration

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 2,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4
    }
  },
  "fp16": {
    "enabled": false
  }
}
```

**Note:** FP16 is disabled for numerical stability in MCMC. The log probability computations require full precision.

## Running with SLURM

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=bayesian_nn

source ~/myenv/bin/activate
deepspeed --num_gpus=2 parallel_tempering_mcmc.py
```

## Expected Output

```
Parallel Tempering MCMC with 2 GPUs

GPU 0: Temperature = 1.00 (cold chain)
GPU 1: Temperature = 4.00 (hot chain)

Iteration 100:
  Chain 0 acceptance: 0.32
  Chain 1 acceptance: 0.45
  Swap attempts: 10, accepted: 3

Iteration 1000:
  Collected 500 posterior samples from cold chain
  Mean prediction uncertainty: 0.15

Final Results:
  Posterior mean predictions: [...]
  95% credible intervals: [...]
```

---

## Why Multiple GPUs for Bayesian Inference?

The connection between parallel tempering and multi-GPU computing is natural:

```mermaid
graph TB
    subgraph "Multi-GPU Parallel Tempering"
        GPU0["GPU 0<br/>T = 1.0<br/>Cold Chain<br/>(Collect Samples)"]
        GPU1["GPU 1<br/>T = 2.0<br/>Warm Chain"]
        GPU2["GPU 2<br/>T = 4.0<br/>Hot Chain"]
        GPU3["GPU 3<br/>T = 8.0<br/>Very Hot Chain<br/>(Free Exploration)"]

        GPU0 <-->|"Swap"| GPU1
        GPU1 <-->|"Swap"| GPU2
        GPU2 <-->|"Swap"| GPU3
    end

    style GPU0 fill:#1b4a75,stroke:#5590c0,stroke-width:1.5px,color:#ffffff
    style GPU3 fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
```

| # GPUs | Temperature Range | Benefit |
|--------|-------------------|---------|
| 2 | T ∈ {1, 4} | Basic exploration |
| 4 | T ∈ {1, 2, 4, 8} | Better mode discovery |
| 8 | T ∈ {1, 1.5, 2, 3, 4, 6, 8, 12} | Fine-grained ladder, high swap rates |

**More GPUs = More Temperatures = Better Posterior Exploration**

---

## Summary: Key Equations

### Bayes' Theorem
$$
P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) \cdot P(\theta)}{P(\mathcal{D})}
$$

### Tempered Posterior
$$
P_T(\theta | \mathcal{D}) \propto P(\mathcal{D} | \theta)^{1/T} \cdot P(\theta)
$$

### Metropolis-Hastings Acceptance
$$
\alpha = \min\left(1, \frac{P(\theta^* | \mathcal{D})}{P(\theta^{(t)} | \mathcal{D})}\right)
$$

### Swap Acceptance
$$
\Delta = \left(\frac{1}{T_i} - \frac{1}{T_j}\right) \cdot \left(\log P(\mathcal{D} | \theta_j) - \log P(\mathcal{D} | \theta_i)\right)
$$

### Predictive Distribution
$$
P(y | x, \mathcal{D}) \approx \frac{1}{N} \sum_{i=1}^{N} P(y | x, \mathbf{w}^{(i)})
$$

---

## Use Cases

- **Uncertainty estimation**: Get confidence intervals on predictions
- **Model selection**: Compare models via marginal likelihood
- **Robust predictions**: Average over parameter uncertainty
- **Scientific inference**: Proper uncertainty propagation
- **Safety-critical applications**: Know when the model is uncertain

## Troubleshooting

### Low Acceptance Rate

- Reduce step size in proposals
- Increase temperature range
- Check log posterior computation

### Poor Mixing

- Add more temperatures (use more GPUs)
- Increase swap frequency
- Adjust temperature ladder spacing

### Low Swap Acceptance

- Use geometric temperature spacing
- Reduce temperature ratio between adjacent chains
- Ensure log likelihood computation is correct

---

## How This Compares to Other Bayesian Deep Learning Methods

MCMC with parallel tempering is the *asymptotically exact* option — given enough compute it samples the true posterior. It is also by far the most expensive. Knowing the alternatives clarifies what you are buying.

```mermaid
flowchart TB
    POST["The target: p(theta | D)<br/>intractable normalizer"]

    subgraph EXACT["Asymptotically exact — sample the posterior"]
        direction TB
        MCMC["MCMC / parallel tempering<br/>THIS TUTORIAL<br/>exact in the limit, very expensive"]
        SGMCMC["SG-MCMC — SGLD, SGHMC<br/>minibatch gradients<br/>scales, but biased at finite step size"]
    end

    subgraph APPROX["Approximate — fit a simpler family"]
        direction TB
        VI["Variational inference<br/>Bayes by Backprop<br/>fast, mode-seeking, underestimates variance"]
        LAP["Laplace approximation<br/>Gaussian at the MAP<br/>nearly free post-hoc"]
        MCD["MC Dropout<br/>dropout at test time<br/>cheapest, weakest guarantees"]
        ENS["Deep ensembles<br/>N independent trainings<br/>not Bayesian, usually best-calibrated"]
    end

    POST --> MCMC
    POST --> SGMCMC
    POST --> VI
    POST --> LAP
    POST --> MCD
    POST --> ENS

    classDef deep fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
    classDef base fill:#16324f,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
    classDef steel fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff
    classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
    class POST bright
    class MCMC,SGMCMC steel
    class VI,LAP,MCD,ENS base
    class EXACT,APPROX deep
```

| Method | Cost vs. one training run | Captures multimodality | Notes |
|---|---|---|---|
| Parallel tempering MCMC | $10^2$–$10^4\times$ | **Yes** — the point of the method | Exact in the limit; needs $O(\sqrt{d})$ rungs |
| SG-MCMC (SGLD/SGHMC) | $2$–$10\times$ | Partially, with cyclical step sizes | Minibatch noise biases the stationary distribution |
| Variational inference | $2$–$3\times$ | No — mean-field is unimodal | Minimizes $D_{\mathrm{KL}}(q\|p)$, which is **mode-seeking** and systematically under-covers |
| Laplace approximation | $\approx 1\times$ + curvature | No | Post-hoc on a trained net; only needs a Hessian approximation |
| MC Dropout | $\approx 1\times$ | No | Interpretable as VI with a very restrictive $q$; cheap but poorly calibrated |
| Deep ensembles | $N\times$ | **In practice, yes** | Not formally Bayesian, but repeatedly the strongest baseline |

:::note Deep ensembles are the honest baseline
Lakshminarayanan et al. (2017) showed that simply training $N$ networks from different random initializations and averaging their predictions matches or beats most principled Bayesian approximations on calibration and out-of-distribution detection. Independent initializations land in genuinely different modes, so an ensemble captures the multimodality that mean-field VI cannot — which is arguably why it works (Wilson & Izmailov, 2020, argue it is *better* understood as approximate Bayesian marginalization than as a non-Bayesian trick).

The practical implication for this tutorial: parallel tempering is worth its cost when you need **calibrated posterior samples** — credible intervals with coverage guarantees, decomposition of epistemic and aleatoric uncertainty, small-data regimes where the prior genuinely matters. If you only need good predictive uncertainty on a large dataset, train five networks and average. Be clear about which problem you have.
:::

:::warning The cold posterior effect
Wenzel et al. (2020) reported that BNNs frequently predict *better* when the posterior is artificially sharpened — sampling from $p(\theta \mid \mathcal{D})^{1/T}$ with $T < 1$ — than at the true Bayes posterior $T = 1$. Taken at face value this is uncomfortable: exact Bayesian inference underperforming a deliberately wrong tempering.

Subsequent work locates the cause in the modelling assumptions rather than in Bayes. Aitchison (2021) attributes it largely to data augmentation and curation making the effective likelihood mis-specified, and Fortuin et al. (2022) show much of the effect disappears under better-chosen (heavy-tailed, correlated) priors than the default isotropic Gaussian.

For this page the point is practical: if your $T=1$ chain is well-mixed and still predicts worse than a plain MAP estimate, suspect the **prior and likelihood specification** before suspecting the sampler. Note also that the cold-posterior $T$ is the same $T$ as the tempering ladder — the $T=1$ rung is the one you draw inference from, and the rest exist only to help it mix.
:::

## Next Steps

- [Stock Prediction](/docs/tutorials/intermediate/stock-prediction) - Real-world application
- [HuggingFace Overview](/docs/tutorials/huggingface/overview) - Large model training
- [Basic Neural Network](/docs/tutorials/basic/neural-network#3-loss-functions-what-they-assume-and-when-they-fail) - losses as likelihoods, the frequentist counterpart to this page

## References

**Bayesian inference and MCMC**

1. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. *J. Chemical Physics*, 21(6), 1087–1092.
2. Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97–109.
3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
4. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In *Handbook of Markov Chain Monte Carlo*. [arXiv:1206.1901](https://arxiv.org/abs/1206.1901)
5. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv:1701.02434](https://arxiv.org/abs/1701.02434)

**Parallel tempering / replica exchange**

6. Swendsen, R. H., & Wang, J.-S. (1986). Replica Monte Carlo Simulation of Spin-Glasses. *Physical Review Letters*, 57(21), 2607–2609. — the original method.
7. Geyer, C. J. (1991). Markov Chain Monte Carlo Maximum Likelihood. *Computing Science and Statistics: Proc. 23rd Symposium on the Interface*. — introduces it to statistics.
8. Earl, D. J., & Deem, M. W. (2005). Parallel tempering: Theory, applications, and new perspectives. *Phys. Chem. Chem. Phys.*, 7, 3910–3916. — the standard review.
9. Kone, A., & Kofke, D. A. (2005). Selection of temperature intervals for parallel-tempering simulations. *J. Chemical Physics*, 122(20), 206101. — the ~0.23 acceptance target.
10. Atchadé, Y. F., Roberts, G. O., & Rosenthal, J. S. (2011). Towards optimal scaling of Metropolis-coupled Markov chain Monte Carlo. *Statistics and Computing*, 21(4), 555–568.

**Convergence diagnostics**

11. Gelman, A., & Rubin, D. B. (1992). Inference from Iterative Simulation Using Multiple Sequences. *Statistical Science*, 7(4), 457–472. — $\hat R$.
12. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved $\hat{R}$ for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718. [arXiv:1903.08008](https://arxiv.org/abs/1903.08008)

**Bayesian neural networks**

13. MacKay, D. J. C. (1992). A Practical Bayesian Framework for Backpropagation Networks. *Neural Computation*, 4(3), 448–472.
14. Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer. — HMC for BNNs; the infinite-width/GP correspondence.
15. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *ICML 2015*. [arXiv:1505.05424](https://arxiv.org/abs/1505.05424) — Bayes by Backprop.
16. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML 2016*. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
17. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS 2017*. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
18. Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. *NeurIPS 2020*. [arXiv:2002.08791](https://arxiv.org/abs/2002.08791)
19. Izmailov, P., Vikram, S., Hoffman, M. D., & Wilson, A. G. (2021). What Are Bayesian Neural Network Posteriors Really Like? *ICML 2021*. [arXiv:2104.14421](https://arxiv.org/abs/2104.14421) — full-batch HMC as a gold-standard reference.
20. Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS 2017*. [arXiv:1703.04977](https://arxiv.org/abs/1703.04977) — the aleatoric/epistemic decomposition.

**Scalable and tempered posteriors**

21. Welling, M., & Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. *ICML 2011*. — SGLD.
22. Chen, T., Fox, E. B., & Guestrin, C. (2014). Stochastic Gradient Hamiltonian Monte Carlo. *ICML 2014*. [arXiv:1402.4102](https://arxiv.org/abs/1402.4102)
23. Zhang, R., Li, C., Zhang, J., Chen, C., & Wilson, A. G. (2020). Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning. *ICLR 2020*. [arXiv:1902.03932](https://arxiv.org/abs/1902.03932)
24. Wenzel, F., Roth, K., Veeling, B. S., et al. (2020). How Good is the Bayes Posterior in Deep Neural Networks Really? *ICML 2020*. [arXiv:2002.02405](https://arxiv.org/abs/2002.02405) — the cold posterior effect.
25. Aitchison, L. (2021). A statistical theory of cold posteriors in deep neural networks. *ICLR 2021*. [arXiv:2008.05912](https://arxiv.org/abs/2008.05912)
26. Fortuin, V., Garriga-Alonso, A., Ober, S. W., et al. (2022). Bayesian Neural Network Priors Revisited. *ICLR 2022*. [arXiv:2102.06571](https://arxiv.org/abs/2102.06571)

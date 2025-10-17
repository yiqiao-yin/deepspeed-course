# Multi-Agent LLM Training with GRPO

Train multi-agent language models using Group Relative Policy Optimization (GRPO) on mathematical reasoning tasks with diverse instruction variants and reward strategies.

## Features

- 🤖 **Multi-Agent Architecture**: Multiple agents with diverse instruction prompts
- 🎯 **GRPO Algorithm**: Group Relative Policy Optimization for efficient RL training
- 🧮 **Math Reasoning**: Trained on GSM8K-style mathematical word problems
- 🔄 **Hidden State Aggregation**: Ensemble learning via hidden state averaging
- 🏆 **Custom Rewards**: Two reward strategies (unique chars vs. similarity-based)
- 📊 **Chain-of-Thought**: Incorporates reasoning steps in training
- ⚡ **Small Model**: Qwen-1.5B - efficient training on single GPU

## Project Overview

This project demonstrates two approaches to training multi-agent language models using GRPO:

### 1. **main.py** - Synthetic Dataset with Simple Reward
### 2. **train_grpo_math.py** - Real GSM8K Dataset with Similarity Reward

Both scripts implement a **MultiAgentLLM** framework that:
1. Generates multiple agent outputs with diverse instruction variants
2. Aggregates hidden states across agents
3. Trains using GRPO with custom reward functions

---

## Motivation

### Why Multi-Agent LLMs?

Traditional single-agent LLMs may get stuck in local optima or exhibit limited reasoning diversity. Multi-agent systems address this by:

1. **Diverse Perspectives**: Different instruction prompts encourage varied reasoning paths
2. **Ensemble Benefits**: Aggregating multiple outputs improves robustness
3. **Exploration**: Multiple agents explore the solution space more thoroughly
4. **Consensus Building**: Hidden state averaging creates stronger representations

### Why GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm designed for language model training:

- **Sample Efficiency**: More efficient than PPO for text generation tasks
- **Stability**: Relative rewards reduce variance in policy updates
- **Simplicity**: Easier to implement than full RLHF pipelines
- **Group-wise Learning**: Learns from groups of completions simultaneously

---

## Mathematical Foundations

### GRPO Algorithm

GRPO optimizes the language model policy $\pi_\theta$ by maximizing expected rewards while staying close to a reference policy $\pi_{\text{ref}}$.

#### Objective Function

The GRPO objective combines reward maximization with a KL-divergence penalty:

$$
\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right]
$$

Where:
- $x$ = input prompt
- $y$ = generated completion
- $r(x, y)$ = reward function
- $\beta$ = KL penalty coefficient
- $\pi_\theta$ = current policy (model being trained)
- $\pi_{\text{ref}}$ = reference policy (frozen base model)

#### Group Relative Rewards

GRPO computes rewards **relative to the group average**, reducing variance:

$$
\tilde{r}_i = r_i - \frac{1}{G} \sum_{j=1}^{G} r_j
$$

Where:
- $r_i$ = raw reward for completion $i$
- $G$ = group size (number of completions per prompt)
- $\tilde{r}_i$ = normalized reward

This normalization ensures that:
1. Rewards are centered around zero
2. The model learns from relative performance
3. Training is more stable across different reward scales

#### Policy Gradient Update

The policy parameters $\theta$ are updated using the gradient:

$$
\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{x, y} \left[ \tilde{r}(x, y) \cdot \nabla_\theta \log \pi_\theta(y|x) - \beta \cdot \nabla_\theta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

In practice, this is approximated using sampled completions:

$$
\nabla_\theta \mathcal{J}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left[ \tilde{r}_i \cdot \nabla_\theta \log \pi_\theta(y_i|x_i) \right]
$$

### Multi-Agent Hidden State Aggregation

The multi-agent framework aggregates hidden states from $K$ agents:

$$
\mathbf{H}_{\text{agg}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{H}_k
$$

Where:
- $\mathbf{H}_k \in \mathbb{R}^{L \times d}$ = hidden states from agent $k$
- $L$ = sequence length
- $d$ = hidden dimension
- $\mathbf{H}_{\text{agg}}$ = aggregated representation

To handle variable-length sequences, hidden states are padded:

$$
\mathbf{H}_k^{\text{pad}} = \begin{cases}
\mathbf{H}_k & \text{if } L_k = L_{\max} \\
[\mathbf{H}_k; \mathbf{0}_{(L_{\max} - L_k) \times d}] & \text{otherwise}
\end{cases}
$$

### Reward Functions

#### 1. Unique Character Reward (main.py)

A simple diversity reward:

$$
r_{\text{unique}}(y) = |\{ c : c \in y \}|
$$

Where $|\cdot|$ denotes set cardinality. This encourages diverse token generation.

#### 2. Similarity Reward (train_grpo_math.py)

Measures string similarity to reference answer:

$$
r_{\text{sim}}(y, y_{\text{ref}}) = 100 \cdot \text{SequenceMatcher}(y, y_{\text{ref}})
$$

The SequenceMatcher computes longest common subsequence (LCS) ratio:

$$
\text{SequenceMatcher}(y, y_{\text{ref}}) = \frac{2 \cdot |LCS(y, y_{\text{ref}})|}{|y| + |y_{\text{ref}}|}
$$

Where:
- $|LCS(y, y_{\text{ref}})|$ = length of longest common subsequence
- $|y|, |y_{\text{ref}}|$ = lengths of predicted and reference strings

---

## Comparison of Two Scripts

| Aspect | **main.py** | **train_grpo_math.py** |
|--------|-------------|------------------------|
| **Dataset** | Synthetic (20 math problems) | Real GSM8K (8000+ problems) |
| **Dataset Source** | Hardcoded in script | Loaded from HuggingFace Hub |
| **Reward Function** | Unique characters count | String similarity to reference |
| **Prompt Structure** | 4 instruction variants | 4 instruction variants (randomized) |
| **Includes CoT** | Yes (separate "think" field) | Yes (from dataset "cot" field) |
| **Training Samples** | 20 samples | 1000 samples (configurable) |
| **Reward Scale** | ~10-40 | 0-100 (percentage) |
| **Use Case** | Educational/prototyping | Production-ready training |
| **Complexity** | Simpler, self-contained | More robust, scalable |

### Detailed Differences

#### Dataset Structure

**main.py**:
```python
{
    "question": "Natalia sold clips to 48 of her friends...",
    "variants": ["<instruction>Solve step by step...</instruction><question>...",
                 "<instruction>Use short arithmetic...</instruction><question>...",
                 ...],
    "answer": "48 in April, 24 in May. Total: 72",
    "think": "She sold half in May, so divide and add both months."
}
```

**train_grpo_math.py**:
```python
{
    "question": "Natalia sold clips to 48 of her friends...",
    "cot": "She sold half in May, so divide and add both months.",
    "answer": "72"
}
```

#### Reward Philosophy

**main.py**:
- Dummy reward (unique characters)
- Purpose: Demonstrate GRPO mechanics
- Not task-specific

**train_grpo_math.py**:
- Similarity-based reward
- Purpose: Guide model toward correct answers
- Task-specific and interpretable

#### Training Scale

**main.py**:
- 20 handcrafted problems
- Quick prototyping
- ~1 minute training

**train_grpo_math.py**:
- 1000+ problems from GSM8K
- Production-ready
- ~10-30 minutes training

---

## Quick Start

### 1. Initial Setup

Start with a fresh RunPod instance (recommend >= 1x RTX 4090 or A100):

```bash
# Install uv package manager
pip install uv

# Initialize new project
uv init multi-agent-grpo
cd multi-agent-grpo

# Add core dependencies
uv add "torch>=2.0.0"
uv add "transformers>=4.55.0"
uv add "trl>=0.20.0"
uv add "datasets>=2.14.0"
uv add "accelerate>=0.24.0"
uv add "huggingface-hub>=0.17.0"

# Development dependencies
uv add --dev "black" "isort" "flake8"
```

### 2. Project Structure

```
multi-agent-grpo/
├── main.py                          # Synthetic dataset + simple reward
├── train_grpo_math.py               # GSM8K dataset + similarity reward
├── multi_agent_train_data/          # Saved dataset (main.py)
├── multi_agent_trained/             # Output: fine-tuned model
├── requirements.txt                 # Generated by uv
└── README.md                        # This file
```

### 3. Environment Configuration

```bash
# Optional: Set Hugging Face token for private models
export HF_TOKEN="your_huggingface_token_here"

# Optional: Configure cache directory
export HF_HOME="/path/to/cache"

# Optional: Enable CUDA optimizations
export CUDA_VISIBLE_DEVICES="0"
```

### 4. Choose Your Training Approach

#### Option A: Synthetic Dataset (main.py)

**Best for**: Learning, prototyping, testing GRPO mechanics

```bash
# Run training
uv run python main.py
```

**What happens**:
1. Creates 20 synthetic math problems
2. Generates 4 instruction variants per problem
3. Saves dataset to `./multi_agent_train_data`
4. Trains with unique character reward
5. Saves model to `./multi_agent_trained`

**Expected output**:
```
✅ Training complete. Model saved to ./multi_agent_trained
```

#### Option B: GSM8K Dataset (train_grpo_math.py)

**Best for**: Production training, better math performance

```bash
# Run training
uv run python train_grpo_math.py
```

**What happens**:
1. Downloads GSM8K dataset from HuggingFace
2. Samples 1000 training examples
3. Generates random instruction variants
4. Trains with similarity-based reward
5. Saves model to `./multi_agent_trained`

**Expected output**:
```
🚀 Loading dataset...
🤖 Initializing Multi-Agent LLM...
🎯 Training with GRPO...
✅ Training complete. Model saved to ./multi_agent_trained
```

---

## Understanding the Code

### Core Components

#### 1. MultiAgentLLM Class

The main class that orchestrates multi-agent training:

```python
class MultiAgentLLM:
    def __init__(self, model_name, num_agents=4)
    def generate_agent_outputs(self, prompt_variants)
    def aggregate_hidden_states(self, agent_outputs)
    def train_grpo(self, dataset)
```

**Responsibilities**:
- Load base model with value head (for RL)
- Generate completions from multiple agents
- Aggregate hidden states
- Train using GRPO

#### 2. Stopping Criteria (Lines 12-22)

Custom stopping condition to halt generation at special tokens:

```python
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -len(token):].tolist() == token
                   for token in self.stop_token_ids)
```

**Purpose**: Stop generation when `</response>` token is encountered

#### 3. Agent Output Generation (Lines 41-51 in main.py)

Each agent generates a completion for its variant:

```python
def generate_agent_outputs(self, prompt_variants):
    outputs = []
    for prompt in prompt_variants:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            stopping_criteria=self.stopping_criteria
        )
        outputs.append(output[0])
    return outputs
```

**Flow**:
1. For each prompt variant (different instruction)
2. Tokenize and generate completion
3. Stop at `</response>` or max_new_tokens
4. Collect all agent outputs

#### 4. Hidden State Aggregation (Lines 53-71 in main.py)

Average hidden states across agents:

```python
def aggregate_hidden_states(self, agent_outputs):
    # Extract hidden states
    hidden_states = []
    for output_ids in agent_outputs:
        out = transformer(input_ids=output_ids)
        hidden_states.append(out.last_hidden_state)

    # Pad to same length
    max_len = max(h.shape[1] for h in hidden_states)
    padded = [pad_to_length(h, max_len) for h in hidden_states]

    # Average across agents
    return torch.stack(padded).mean(dim=0)
```

**Mathematical Operation**:
$$
\mathbf{H}_{\text{ensemble}} = \frac{1}{K} \sum_{k=1}^{K} \text{Pad}(\mathbf{H}_k, L_{\max})
$$

#### 5. GRPO Training (Lines 73-99 in main.py)

Main training loop:

```python
def train_grpo(self, dataset):
    # Format dataset
    prompts = [sample["variants"][0] for sample in dataset]
    completions = [sample["answer"] for sample in dataset]

    formatted_dataset = Dataset.from_dict({
        "prompt": prompts,
        "completion": completions
    })

    # Initialize trainer
    trainer = GRPOTrainer(
        model=self.model_name,
        reward_funcs=reward_unique_chars,
        train_dataset=formatted_dataset
    )

    # Train and save
    trainer.train()
    trainer.model.save_pretrained("./multi_agent_trained")
```

**Key Steps**:
1. Flatten dataset to prompt/completion pairs
2. Create HuggingFace Dataset object
3. Initialize GRPOTrainer with reward function
4. Run training loop
5. Save fine-tuned model

### Reward Functions Deep Dive

#### Unique Character Reward (main.py:22-24)

```python
def reward_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]
```

**Analysis**:
- Simple diversity metric
- Range: ~10-40 for typical text
- Encourages vocabulary variety
- Not task-specific

**Example**:
```
"72" → {7, 2} → reward = 2
"The answer is 72" → {T, h, e, , a, n, s, w, r, i, 7, 2} → reward = 13
```

#### Similarity Reward (train_grpo_math.py:25-35)

```python
def make_similarity_reward_fn(references):
    def reward_fn(completions, **kwargs):
        rewards = []
        for pred, ref in zip(completions, references):
            similarity = SequenceMatcher(None, pred.strip(), ref.strip()).ratio()
            rewards.append(similarity * 100)
        return rewards
    return reward_fn
```

**Analysis**:
- Task-specific (math correctness)
- Range: 0-100 (percentage)
- Measures longest common subsequence
- Guides toward correct answers

**Example**:
```
Reference: "72"
Prediction: "72" → LCS = "72" → ratio = 1.0 → reward = 100
Prediction: "The answer is 72" → LCS = "72" → ratio = 0.21 → reward = 21
Prediction: "48" → LCS = "" → ratio = 0.0 → reward = 0
```

### Instruction Variants

Both scripts use diverse instruction prompts to encourage different reasoning styles:

**main.py (Lines 103-108)**:
```python
instruction_variants = [
    "This is a math problem. Solve it step by step.",
    "This is a math problem. Use short arithmetic first.",
    "This is a math problem. Focus on final answer.",
    "This is a math problem. Use chain of thought."
]
```

**train_grpo_math.py (Lines 93-98)**:
```python
instructions = [
    "Solve step by step.",
    "Use chain of thought.",
    "Be concise but correct.",
    "Explain then answer."
]
```

**Rationale**: Different instructions elicit different reasoning patterns, creating a diverse multi-agent ensemble.

---

## Training Process

### Step-by-Step Training Flow

#### Phase 1: Initialization

1. **Load Base Model**:
   ```python
   model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
   ```
   - Downloads Qwen-1.5B (~3GB)
   - Adds value head for RL training
   - Initializes tokenizer

2. **Prepare Dataset**:
   - **main.py**: Creates synthetic problems
   - **train_grpo_math.py**: Downloads GSM8K

3. **Define Stopping Criteria**:
   - Encodes `</response>` token
   - Creates StoppingCriteria object

#### Phase 2: Data Formatting

1. **Create Prompt Variants**:
   ```python
   prompt = f"<instruction>{instruction}</instruction><question>{question}</question>"
   ```

2. **Format for GRPO**:
   ```python
   formatted_dataset = {
       "prompt": [...],
       "completion": [...]
   }
   ```

#### Phase 3: GRPO Training Loop

For each training iteration:

1. **Sample Batch**: Select prompts from dataset

2. **Generate Completions**:
   $$
   y_1, y_2, \ldots, y_G \sim \pi_\theta(\cdot | x)
   $$

3. **Compute Rewards**:
   $$
   r_1, r_2, \ldots, r_G = R(y_1), R(y_2), \ldots, R(y_G)
   $$

4. **Normalize Rewards**:
   $$
   \tilde{r}_i = r_i - \frac{1}{G} \sum_{j=1}^{G} r_j
   $$

5. **Compute Policy Gradient**:
   $$
   g = \nabla_\theta \mathbb{E}_{y \sim \pi_\theta} [\tilde{r}(y) \log \pi_\theta(y|x)]
   $$

6. **Update Parameters**:
   $$
   \theta \leftarrow \theta + \alpha \cdot g
   $$

7. **Apply KL Penalty** (if diverging from reference):
   $$
   \theta \leftarrow \theta - \beta \cdot \nabla_\theta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
   $$

#### Phase 4: Model Saving

1. **Save Fine-tuned Weights**:
   ```python
   trainer.model.save_pretrained("./multi_agent_trained")
   ```

2. **Save Tokenizer**:
   ```python
   trainer.tokenizer.save_pretrained("./multi_agent_trained")
   ```

---

## Using the Trained Model

### Load Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./multi_agent_trained")
model = AutoModelForCausalLM.from_pretrained("./multi_agent_trained")
```

### Single-Agent Inference

```python
prompt = "<instruction>Solve step by step.</instruction><question>A bakery made 240 cookies. If they packed 12 cookies in each box, how many boxes did they fill?</question>"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
# Expected: "240 / 12 = 20 </response>"
```

### Multi-Agent Inference (Ensemble)

```python
instructions = [
    "Solve step by step.",
    "Use chain of thought.",
    "Be concise but correct.",
    "Explain then answer."
]

question = "A bakery made 240 cookies. If they packed 12 cookies in each box, how many boxes did they fill?"

responses = []
for instruction in instructions:
    prompt = f"<instruction>{instruction}</instruction><question>{question}</question>"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    responses.append(response)

# Vote or average results
print("Agent responses:", responses)
```

### Extract Final Answer

```python
def extract_answer(response):
    """Extract numerical answer from model response."""
    import re
    # Look for numbers in response
    numbers = re.findall(r'\d+', response)
    return numbers[-1] if numbers else None

answer = extract_answer(response)
print(f"Final answer: {answer}")
```

---

## Configuration and Hyperparameters

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | Qwen-1.5B | Small efficient model |
| Model Type | CausalLM + ValueHead | For RL training |
| Max New Tokens | 64 | Generation limit |
| Stopping Token | `</response>` | Custom stop sequence |

### GRPO Hyperparameters

Default values in TRL GRPOTrainer:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate | 5e-6 | Adam optimizer LR |
| Batch Size | 8 | Samples per update |
| KL Penalty (β) | 0.01 | KL divergence weight |
| Group Size | 4 | Completions per prompt |
| Gamma | 0.99 | Discount factor |

### Custom Configuration

You can override defaults:

```python
from trl import GRPOConfig

config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    kl_penalty="kl",  # or "abs", "mse"
    kl_coef=0.02,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
)

trainer = GRPOTrainer(
    model=model_name,
    train_dataset=dataset,
    reward_funcs=reward_fn,
    config=config
)
```

---

## Advanced Topics

### 1. Custom Reward Functions

Create domain-specific rewards:

```python
def math_accuracy_reward(completions, ground_truth, **kwargs):
    """Reward based on exact numerical match."""
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        # Extract numbers from completion
        pred_nums = extract_numbers(comp)
        true_nums = extract_numbers(truth)

        # Check if any predicted number matches
        if any(p == t for p in pred_nums for t in true_nums):
            rewards.append(100.0)
        else:
            rewards.append(0.0)
    return rewards
```

### 2. Multi-Turn Conversations

Extend to dialogue:

```python
conversation = [
    {"role": "user", "content": "Solve: 2 + 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Now multiply by 3"},
]

prompt = format_conversation(conversation)
```

### 3. Hybrid Rewards

Combine multiple reward signals:

```python
def hybrid_reward(completions, references, **kwargs):
    # Similarity reward
    sim_rewards = similarity_reward(completions, references)

    # Diversity reward
    div_rewards = [len(set(c)) for c in completions]

    # Combine (weighted sum)
    return [0.8 * s + 0.2 * d for s, d in zip(sim_rewards, div_rewards)]
```

### 4. Curriculum Learning

Gradually increase difficulty:

```python
# Start with easy problems
dataset_easy = dataset.filter(lambda x: x["difficulty"] == "easy")
agent_model.train_grpo(dataset_easy, num_samples=500)

# Then medium problems
dataset_medium = dataset.filter(lambda x: x["difficulty"] == "medium")
agent_model.train_grpo(dataset_medium, num_samples=500)

# Finally hard problems
dataset_hard = dataset.filter(lambda x: x["difficulty"] == "hard")
agent_model.train_grpo(dataset_hard, num_samples=500)
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:
```python
# Reduce batch size
config = GRPOConfig(batch_size=4)

# Use gradient accumulation
config = GRPOConfig(
    batch_size=4,
    gradient_accumulation_steps=4  # Effective batch = 16
)

# Use smaller model
model_name = "Qwen/Qwen-0.5B"  # Instead of 1.5B
```

#### Slow Training

**Symptoms**: Training takes hours

**Solutions**:
```python
# Reduce number of samples
agent_model.train_grpo(dataset, num_samples=100)  # Instead of 1000

# Reduce generation length
output = model.generate(max_new_tokens=32)  # Instead of 64

# Use mixed precision
config = GRPOConfig(fp16=True)
```

#### Poor Reward Signal

**Symptoms**: Model doesn't improve, rewards stay flat

**Solutions**:
```python
# Scale rewards appropriately
def scaled_reward(completions, **kwargs):
    raw = similarity_reward(completions)
    return [r * 10 for r in raw]  # Scale up

# Add reward shaping
def shaped_reward(completions, **kwargs):
    base_reward = similarity_reward(completions)
    # Bonus for correct format
    format_bonus = [10 if "</response>" in c else 0 for c in completions]
    return [b + f for b, f in zip(base_reward, format_bonus)]
```

#### Model Divergence

**Symptoms**: Loss becomes NaN, outputs become gibberish

**Solutions**:
```python
# Increase KL penalty
config = GRPOConfig(kl_coef=0.05)  # Instead of 0.01

# Reduce learning rate
config = GRPOConfig(learning_rate=1e-6)  # Instead of 5e-6

# Clip gradients
config = GRPOConfig(max_grad_norm=1.0)
```

#### Dataset Format Errors

**Symptoms**: KeyError or ValueError during training

**Solutions**:
```python
# Validate dataset structure
print(dataset[0])
assert "prompt" in dataset.column_names
assert "completion" in dataset.column_names

# Check for None values
dataset = dataset.filter(lambda x: x["prompt"] is not None)
```

---

## Performance Metrics

### Training Metrics

Monitor during training:

| Metric | Good Range | Interpretation |
|--------|------------|----------------|
| **Training Loss** | Decreasing | Model is learning |
| **Reward Mean** | Increasing | Better completions |
| **Reward Std** | Moderate | Diverse exploration |
| **KL Divergence** | 0.01-0.1 | Policy stability |
| **Generation Length** | Stable | Consistent outputs |

### Evaluation Metrics

For math reasoning:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Exact Match** | Correct final answer | answer == ground_truth |
| **Similarity** | String closeness | SequenceMatcher ratio |
| **Perplexity** | Model confidence | exp(-log_prob) |
| **Pass@K** | Success in K tries | any(tries[:k]) |

### Example Evaluation

```python
def evaluate_model(model, tokenizer, test_dataset):
    correct = 0
    total = 0

    for sample in test_dataset:
        prompt = format_prompt(sample["question"])
        outputs = model.generate(...)
        response = tokenizer.decode(outputs[0])

        predicted = extract_answer(response)
        ground_truth = extract_answer(sample["answer"])

        if predicted == ground_truth:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy
```

---

## System Requirements

### Minimum Requirements
- **GPU**: 1x RTX 3060 (12GB VRAM)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space (model + dataset + checkpoints)
- **CUDA**: 11.8+
- **Internet**: Required for model and dataset downloads

### Recommended Setup
- **GPU**: 1x RTX 4090 (24GB) or A100 (40GB)
- **RAM**: 32GB system RAM
- **Storage**: 100GB SSD
- **CUDA**: 12.1+

### Performance Estimates

| Setup | Training Time (1000 samples) | Memory Usage |
|-------|------------------------------|--------------|
| RTX 3060 (12GB) | ~45 minutes | 10-11GB VRAM |
| RTX 4090 (24GB) | ~20 minutes | 12-14GB VRAM |
| A100 (40GB) | ~15 minutes | 14-16GB VRAM |

---

## References and Further Reading

### Papers

1. **GRPO Algorithm**:
   - "Group Relative Policy Optimization" (DeepMind, 2023)
   - Key insight: Relative rewards for stable RL

2. **PPO (Foundation)**:
   - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
   - Original algorithm GRPO builds upon

3. **RLHF**:
   - Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
   - Broader context for RL in LLMs

4. **GSM8K Dataset**:
   - Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021)
   - Original math reasoning benchmark

### Libraries

- **TRL (Transformer Reinforcement Learning)**: https://github.com/huggingface/trl
- **Transformers**: https://github.com/huggingface/transformers
- **Datasets**: https://github.com/huggingface/datasets

### Related Work

- **Multi-Agent Systems**: Ensemble methods in NLP
- **Chain-of-Thought**: Wei et al. (2022)
- **Reward Modeling**: Learning from human preferences

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test both scripts
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for TRL library
- Alibaba for Qwen model series
- OpenAI for GSM8K dataset inspiration
- DeepMind for GRPO algorithm

---

**Note**: This multi-agent GRPO framework demonstrates cutting-edge reinforcement learning for language models. The dual-script approach allows you to start with simple synthetic data (main.py) and scale to production-ready training (train_grpo_math.py). Perfect for learning RL-based LLM training with mathematical reasoning tasks.

---
sidebar_position: 6
---

# Multi-Agent Training

Multi-agent LLM training for mathematical reasoning with GRPO.

## Overview

This example demonstrates:
- Multiple agents with diverse instructions
- Hidden state aggregation
- Ensemble learning
- GRPO algorithm for RL fine-tuning

**Model:** Qwen-1.5B
**Dataset:** GSM8K-style math problems

## Quick Start

```bash
cd 07_huggingface_trl_multi_agency

# Training
python train_grpo_math.py

# Main entry point
python main.py
```

## Multi-Agent Architecture

```
     ┌─────────────────────────────────────┐
     │           Input Problem             │
     └──────────────┬──────────────────────┘
                    │
     ┌──────────────┼──────────────────────┐
     ▼              ▼                      ▼
┌─────────┐  ┌─────────────┐       ┌─────────────┐
│ Agent 1 │  │  Agent 2    │  ...  │  Agent N    │
│ "Solve  │  │ "Calculate  │       │ "Step by    │
│  step   │  │  carefully" │       │   step"     │
│  by..." │  └──────┬──────┘       └──────┬──────┘
└────┬────┘         │                     │
     │              ▼                     │
     │    ┌────────────────────┐          │
     └───►│ Hidden State Pool  │◄─────────┘
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │    Aggregation     │
          │   (mean/weighted)  │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │   Final Answer     │
          └────────────────────┘
```

## Agent Variants

Each agent receives different instruction prompts:

```python
agent_instructions = [
    "Solve this math problem step by step:",
    "Calculate carefully and show your work:",
    "Break down this problem into smaller parts:",
    "Think through this problem methodically:",
]
```

## GRPO Training

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    num_generations=4,  # Generate 4 responses per prompt
    temperature=0.7,
    learning_rate=5e-5,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_fn=math_reward_function,
)
```

## Reward Function

```python
def math_reward_function(responses, ground_truth):
    rewards = []
    for response in responses:
        # Extract numerical answer
        predicted = extract_answer(response)

        # Compute reward
        if predicted == ground_truth:
            reward = 1.0
        else:
            # Partial credit for close answers
            reward = max(0, 1 - abs(predicted - ground_truth) / ground_truth)

        rewards.append(reward)
    return rewards
```

## Expected Results

| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| GSM8K Accuracy | 45% | 52% |
| Consistency | 0.7 | 0.85 |
| Training Time | 2h | 4h |

## Next Steps

- [Video Training](/docs/tutorials/multimodal/video-text-training) - Multimodal AI
- [Video Speech](/docs/tutorials/multimodal/video-speech-training) - Audio+Video

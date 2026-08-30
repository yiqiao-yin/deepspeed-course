import argparse
import os
import random
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import AutoModelForCausalLM
from trl import GRPOTrainer
import re


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, the run gets as far as loading the model and then dies deep
    inside the training stack -- for this script, with
    "Your setup doesn't support bf16/gpu", which tells a newcomer nothing
    about what went wrong or what to do next. Worse, it happens AFTER the
    model download, so the reader has already waited.

    Set ALLOW_CPU=1 to bypass.
    """
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            You will also need bf16 disabled in the training")
        print("            config, or the trainer raises anyway.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before the run fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  This example runs multi-agent GRPO with TRL. It downloads a")
    print("  1.5B model and needs real GPU memory. On CPU the trainer")
    print("  raises \"Your setup doesn't support bf16/gpu\" — AFTER the")
    print("  download, which is why this check exists.")
    print("\n  No GPU at all? These need none:")
    print("      uv run tests/test_grpo_rewards.py   # the reward logic")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("      ./tests/run_all.sh    # the full logic suite, no downloads")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent one (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 07_huggingface_trl_multi_agency")
    print("      uv run runpod/runpod_ctl.py run 07_huggingface_trl_multi_agency \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria that halts generation when stop tokens are generated."""

    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(input_ids[0, -len(token):].tolist() == token for token in self.stop_token_ids)


def extract_final_answer(text: str):
    """
    Pull the final numeric answer out of a generated solution.

    Handles the GSM8K '#### <answer>' convention and otherwise falls back to
    the last number in the text.

    Returns:
        The answer as a float, or None if no number is present.
    """
    if "####" in text:
        text = text.split("####")[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def reward_answer_correct(completions, **kwargs):
    """
    Verifiable reward: 1.0 if the final answer matches ground truth, else 0.0.

    Replaces the previous string-similarity reward, which had two problems.

    1. It rewarded SURFACE FORM rather than correctness. '42' and '-42' are
       about 95% similar as strings and one of them is wrong; a correct
       solution phrased differently from the reference scored poorly. That is
       precisely the reward-hacking failure that verifiable rewards exist to
       eliminate.

    2. It was MISALIGNED. The old factory closed over the dataset's completion
       list and zipped it positionally against the generated completions. GRPO
       samples G rollouts per prompt, so generation i does not correspond to
       dataset row i and each rollout was scored against the wrong reference.

    Reading references from **kwargs fixes the alignment: GRPOTrainer forwards
    the dataset columns already expanded to match the generated batch.
    """
    references = kwargs.get("completion") or kwargs.get("answer")
    if references is None:
        raise ValueError(
            "reward_answer_correct needs reference answers. Ensure the dataset "
            "has a 'completion' (or 'answer') column."
        )

    rewards = []
    for prediction, reference in zip(completions, references):
        predicted = extract_final_answer(str(prediction))
        expected = extract_final_answer(str(reference))
        rewards.append(
            1.0 if (predicted is not None and expected is not None
                    and abs(predicted - expected) < 1e-6)
            else 0.0
        )
    return rewards


class MultiAgentLLM:
    """
    Multi-agent LLM trainer using GRPO.

    Note on the model class: this previously used
    AutoModelForCausalLMWithValueHead, whose value head is a CRITIC — a PPO
    construct. GRPO replaces the learned baseline with the group mean reward
    and needs no critic; that removal is where its memory saving comes from.
    """

    def __init__(self, model_name: str, num_agents: int = 4):
        self.model_name = model_name
        self.num_agents = num_agents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Define stopping criteria
        stop_sequence = "</response>"
        stop_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens([stop_ids])])

    def generate_agent_outputs(self, prompt_variants):
        """Generate completions from each agent variant."""
        outputs = []
        for prompt in prompt_variants:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                stopping_criteria=self.stopping_criteria
            )
            outputs.append(output[0])
        return outputs

    def aggregate_hidden_states(self, agent_outputs):
        """
        Average final-layer hidden states across agent completions.

        EXPLORATORY analysis utility, not part of training: the GRPO objective
        consumes only per-token log-probabilities and scalar rewards, so no
        hidden state enters it. For genuine ensembling prefer self-consistency
        (sample G chains, majority-vote the final answer).
        """
        hidden_states = []

        with torch.no_grad():
            for output_ids in agent_outputs:
                output_ids = output_ids.unsqueeze(0)
                # output_hidden_states works for any causal LM; the previous
                # `self.model.base_model` access was specific to the value-head
                # wrapper this class no longer uses.
                out = self.model(input_ids=output_ids, output_hidden_states=True)
                last_hidden = out.hidden_states[-1]
                hidden_states.append(last_hidden)

        max_len = max(h.shape[1] for h in hidden_states)
        padded = []
        for h in hidden_states:
            pad_len = max_len - h.shape[1]
            pad_tensor = torch.zeros((1, pad_len, h.shape[2]), dtype=h.dtype, device=h.device)
            padded.append(torch.cat([h, pad_tensor], dim=1))

        return torch.stack(padded).mean(dim=0)

    def train_grpo(self, hf_dataset: Dataset, num_samples: int = 1000,
                   max_steps: int = -1):
        """Train model using GRPO on a formatted dataset."""

        prompts = []
        completions = []

        for sample in hf_dataset.select(range(min(num_samples, len(hf_dataset)))):
            instructions = [
                "Solve step by step.",
                "Use chain of thought.",
                "Be concise but correct.",
                "Explain then answer."
            ]
            instruction = random.choice(instructions)
            prompt = (
                f"<instruction>{instruction}</instruction>"
                f"<question>{sample['question']}</question>"
                f"<think>{sample['cot']}</think>"
            )
            prompts.append(prompt)
            completions.append(sample["answer"] + " </response>")

        formatted_dataset = Dataset.from_dict({
            "prompt": prompts,
            "completion": completions
        })

        # Pass the already-loaded model, not the model NAME. Passing the name
        # made GRPOTrainer load a second copy of the weights, leaving the model
        # built in __init__ untrained and merely resident in memory.
        from trl import GRPOConfig

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=formatted_dataset,
            reward_funcs=reward_answer_correct,
            # max_steps=-1 means "ignore me, use epochs" — Trainer's own
            # convention, so the default reproduces the previous behaviour.
            # It is what makes a cheap dry run possible on a cluster.
            args=GRPOConfig(output_dir="./multi_agent_trained",
                            max_steps=max_steps),
        )

        trainer.train()

        trainer.model.save_pretrained("./multi_agent_trained")
        trainer.tokenizer.save_pretrained("./multi_agent_trained")


if __name__ == "__main__":
    MODEL_ID = "eagle0504/qwen-distilled-scout-1.5b-instruct-gen2"
    DATASET_ID = "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1"

    require_gpu()

    print("🚀 Loading dataset...")
    raw_dataset = load_dataset(DATASET_ID, split="train")

    print("🤖 Initializing Multi-Agent LLM...")
    agent_model = MultiAgentLLM(MODEL_ID)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Stop after this many optimizer steps. -1 means "
                             "no cap. Makes `sbatch run_slurm.sh "
                             "--max-steps 20` a real dry run.")
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_known_args()[0]

    print("🎯 Training with GRPO...")
    agent_model.train_grpo(raw_dataset, num_samples=args.num_samples,
                           max_steps=args.max_steps)

    print("✅ Training complete. Model saved to ./multi_agent_trained")

import os
import random
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
from difflib import SequenceMatcher


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria that halts generation when stop tokens are generated."""

    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(input_ids[0, -len(token):].tolist() == token for token in self.stop_token_ids)


def make_similarity_reward_fn(references):
    """Returns a reward function that compares completions to references using string similarity."""

    def reward_fn(completions, **kwargs):
        rewards = []
        for pred, ref in zip(completions, references):
            similarity = SequenceMatcher(None, pred.strip(), ref.strip()).ratio()
            rewards.append(similarity * 100)  # Scale to 0–100
        return rewards

    return reward_fn


class MultiAgentLLM:
    """Multi-agent LLM trainer using GRPO."""

    def __init__(self, model_name: str, num_agents: int = 4):
        self.model_name = model_name
        self.num_agents = num_agents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

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
        """Average the hidden states across agent completions."""
        transformer = self.model.base_model
        hidden_states = []

        with torch.no_grad():
            for output_ids in agent_outputs:
                output_ids = output_ids.unsqueeze(0)
                out = transformer(input_ids=output_ids)
                last_hidden = out.last_hidden_state
                hidden_states.append(last_hidden)

        max_len = max(h.shape[1] for h in hidden_states)
        padded = []
        for h in hidden_states:
            pad_len = max_len - h.shape[1]
            pad_tensor = torch.zeros((1, pad_len, h.shape[2]), dtype=h.dtype, device=h.device)
            padded.append(torch.cat([h, pad_tensor], dim=1))

        return torch.stack(padded).mean(dim=0)

    def train_grpo(self, hf_dataset: Dataset, num_samples: int = 1000):
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

        reward_fn = make_similarity_reward_fn(completions)

        trainer = GRPOTrainer(
            model=self.model_name,
            train_dataset=formatted_dataset,
            reward_funcs=reward_fn,
        )

        trainer.train()

        trainer.model.save_pretrained("./multi_agent_trained")
        trainer.tokenizer.save_pretrained("./multi_agent_trained")


if __name__ == "__main__":
    MODEL_ID = "eagle0504/qwen-distilled-scout-1.5b-instruct-gen2"
    DATASET_ID = "eagle0504/openai-gsm8k-enhanced-using-together-ai-deepseek-train8k-test1k-v1"

    print("🚀 Loading dataset...")
    raw_dataset = load_dataset(DATASET_ID, split="train")

    print("🤖 Initializing Multi-Agent LLM...")
    agent_model = MultiAgentLLM(MODEL_ID)

    print("🎯 Training with GRPO...")
    agent_model.train_grpo(raw_dataset, num_samples=1000)

    print("✅ Training complete. Model saved to ./multi_agent_trained")

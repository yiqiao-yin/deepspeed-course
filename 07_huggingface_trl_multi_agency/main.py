import os
import torch
import random
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList
)
from trl import GRPOTrainer, AutoModelForCausalLMWithValueHead


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria that halts generation when stop tokens are generated."""
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(input_ids[0, -len(token):].tolist() == token for token in self.stop_token_ids)


def reward_unique_chars(completions, **kwargs):
    """Dummy reward: number of unique characters in the generated output."""
    return [len(set(c)) for c in completions]


class MultiAgentLLM:
    """Multi-agent LLM that performs generation, aggregation, and PPO training."""

    def __init__(self, model_name, num_agents=4):
        self.model_name = model_name
        self.num_agents = num_agents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

        # Define stopping criteria for generation
        stop_sequence = "</response>"
        stop_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens([stop_ids])])

    def generate_agent_outputs(self, prompt_variants):
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

    def train_grpo(self, dataset):
        # Flatten your dataset to match expected format: {prompt, completion}
        prompts = []
        completions = []
        for sample in dataset:
            prompts.append(sample["variants"][0])  # you can random.choice() if you want
            completions.append(sample["answer"])   # target answer
    
        formatted_dataset = {
            "prompt": prompts,
            "completion": completions
        }

        hf_dataset = Dataset.from_dict(formatted_dataset)  # ✅ Correct
        
        # Initialize and run GRPOTrainer
        trainer = GRPOTrainer(
            model=self.model_name,  # e.g. "eagle0504/qwen-distilled-scout-1.5b-instruct-gen2"
            reward_funcs=reward_unique_chars,
            train_dataset=hf_dataset
        )
    
        trainer.train()
    
        # Save model and tokenizer
        trainer.model.save_pretrained("./multi_agent_trained")
        trainer.tokenizer.save_pretrained("./multi_agent_trained")


# Generate sample dataset and save
instruction_variants = [
    "This is a math problem. Solve it step by step.",
    "This is a math problem. Use short arithmetic first.",
    "This is a math problem. Focus on final answer.",
    "This is a math problem. Use chain of thought."
]

questions = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "A bakery made 240 cookies. If they packed 12 cookies in each box, how many boxes did they fill?",
    "Tom read 30 pages on Monday and twice as many on Tuesday. How many pages did he read in total?",
    "A train traveled 150 miles in 3 hours. What was its average speed in miles per hour?",
    "There are 24 apples. If each basket holds 6 apples, how many baskets are needed?",
    "Linda bought 5 packs of pencils. Each pack has 10 pencils. How many pencils did she buy?",
    "Sara had $120. She spent $45 on books. How much money does she have left?",
    "A rectangle has a length of 10 cm and a width of 4 cm. What is its area?",
    "There are 36 students in a class. They are divided equally into 6 groups. How many students are in each group?",
    "Ben has 5 boxes of crayons. Each box contains 8 crayons. How many crayons does he have in total?",
    "A car travels 60 miles per hour. How far will it travel in 4 hours?",
    "A farmer has 72 eggs. He puts 9 eggs in each carton. How many cartons does he use?",
    "Alice has 3 times as many marbles as Bob. If Bob has 12 marbles, how many does Alice have?",
    "There are 100 candies to be divided among 4 kids equally. How many candies does each kid get?",
    "A store sells pencils for $2 each. How much will 7 pencils cost?",
    "If 5 notebooks cost $25, what is the cost of one notebook?",
    "A tree grows 2 feet every year. How tall will it be after 5 years?",
    "A water tank holds 500 liters. If 125 liters are used, how much is left?",
    "You bought 3 shirts at $15 each. How much did you spend?",
    "If you walk 3 miles in one hour, how many miles will you walk in 6 hours?"
]

answers = [
    "48 in April, 24 in May. Total: 72",
    "240 / 12 = 20",
    "30 + 60 = 90",
    "150 / 3 = 50 mph",
    "24 / 6 = 4",
    "5 * 10 = 50",
    "$120 - $45 = $75",
    "10 * 4 = 40",
    "36 / 6 = 6",
    "5 * 8 = 40",
    "60 * 4 = 240",
    "72 / 9 = 8",
    "12 * 3 = 36",
    "100 / 4 = 25",
    "2 * 7 = $14",
    "$25 / 5 = $5",
    "2 * 5 = 10 ft",
    "500 - 125 = 375",
    "3 * 15 = $45",
    "3 * 6 = 18"
]

thinks = [
    "She sold half in May, so divide and add both months.",
    "Divide total cookies by cookies per box.",
    "He read double on Tuesday, then sum both days.",
    "Divide distance by time for speed.",
    "Divide apples by apples per basket.",
    "Multiply packs by pencils per pack.",
    "Subtract amount spent from total.",
    "Multiply length by width.",
    "Divide students into groups.",
    "Multiply boxes by crayons per box.",
    "Multiply speed by time.",
    "Divide total eggs by eggs per carton.",
    "Multiply Bob’s count by 3.",
    "Divide candies by number of kids.",
    "Multiply cost per pencil by quantity.",
    "Divide total cost by quantity.",
    "Multiply yearly growth by number of years.",
    "Subtract used from total.",
    "Multiply quantity by cost per shirt.",
    "Multiply speed by time."
]

samples = []
for i, question in enumerate(questions):
    prompts = [f"<instruction>{instr}</instruction><question>{question}</question>" for instr in instruction_variants]
    samples.append({
        "question": question,
        "variants": prompts,
        "answer": answers[i],
        "think": thinks[i]
    })

train_dataset = Dataset.from_list(samples)
train_dataset.save_to_disk("./multi_agent_train_data")

# Run training
if __name__ == "__main__":
    agent_model = MultiAgentLLM("eagle0504/qwen-distilled-scout-1.5b-instruct-gen2")
    dataset = train_dataset
    agent_model.train_grpo(dataset)
    print("✅ Training complete. Model saved to ./multi_agent_trained")

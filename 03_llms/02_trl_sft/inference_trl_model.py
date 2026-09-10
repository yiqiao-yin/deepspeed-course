"""Inference script for fine-tuned Qwen function calling model.

This script demonstrates how to use the fine-tuned model for function calling
tasks like setting timers and creating reminders.

Usage:
    python inference_trl_model.py --model_path ./sft_qwen_model
    python inference_trl_model.py --model_path ./sft_qwen_model --prompt "Set a timer for 10 minutes"
"""

import argparse
import sys
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_messages(messages: List[Dict[str, str]]) -> str:
    """
    Convert message format to prompt string.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        Formatted prompt string
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            prompt += f"<|tool|>\n{tool_name}: {content}\n"

    prompt += "<|assistant|>\n"  # Let model continue as assistant
    return prompt


def load_model_and_tokenizer(model_path: str):
    """
    Load fine-tuned model and tokenizer.

    Args:
        model_path: Path to the saved model directory

    Returns:
        Tuple of (model, tokenizer, device)
    """
    print(f"🤖 Loading model from: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()  # Set to evaluation mode

        print(f"✅ Model loaded successfully")
        print(f"   - Device: {device}")
        print(f"   - Model dtype: {next(model.parameters()).dtype}")

        return model, tokenizer, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def generate_response(
    model,
    tokenizer,
    device: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 150
) -> str:
    """
    Generate response from the model.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device to run inference on
        messages: List of conversation messages
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated response string
    """
    # Format messages into prompt
    formatted_prompt = format_messages(messages)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Generate
    print("\n🧠 Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return response


def run_sample_inference(model, tokenizer, device: str) -> None:
    """
    Run predefined sample inference examples.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device to run inference on
    """
    # Define sample prompts
    samples = [
        {
            "name": "Timer (5 minutes)",
            "messages": [
                {"role": "user", "content": "Set a timer for 5 minutes."}
            ]
        },
        {
            "name": "Reminder (Medicine)",
            "messages": [
                {"role": "user", "content": "Remind me to take medicine at 9 PM."}
            ]
        },
        {
            "name": "Timer (30 seconds)",
            "messages": [
                {"role": "user", "content": "Set a 30 second timer."}
            ]
        },
        {
            "name": "Reminder (Meeting)",
            "messages": [
                {"role": "user", "content": "Create a reminder for tomorrow's meeting at 2 PM."}
            ]
        }
    ]

    print(f"\n{'='*80}")
    print("🎯 Running Sample Inference Examples")
    print(f"{'='*80}\n")

    for i, sample in enumerate(samples, 1):
        print(f"\n📝 Example {i}/{len(samples)}: {sample['name']}")
        print(f"{'─'*80}")
        print(f"User: {sample['messages'][0]['content']}")

        response = generate_response(
            model,
            tokenizer,
            device,
            sample['messages'],
            max_new_tokens=150
        )

        print(f"\n{'─'*80}")
        print("🤖 Model Response:")
        print(f"{'─'*80}")
        print(response)
        print(f"{'─'*80}")


def run_interactive_inference(model, tokenizer, device: str) -> None:
    """
    Run interactive inference loop.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device to run inference on
    """
    print(f"\n{'='*80}")
    print("💬 Interactive Inference Mode")
    print(f"{'='*80}")
    print("\nType your prompts below (or 'quit' to exit):")
    print("Examples:")
    print("  - Set a timer for 10 minutes")
    print("  - Remind me to call mom at 7 PM")
    print("  - Create a reminder for my dentist appointment tomorrow")
    print("")

    conversation_history = []

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Generate response
        response = generate_response(
            model,
            tokenizer,
            device,
            conversation_history,
            max_new_tokens=150
        )

        print(f"\n🤖 Assistant:\n{response}")

        # Extract assistant response and add to history
        # (In a production system, you'd parse this more carefully)
        conversation_history.append({
            "role": "assistant",
            "content": response
        })

        # Keep conversation history manageable (last 10 turns)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Inference script for fine-tuned Qwen function calling model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./sft_qwen_model",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run inference on (skips interactive mode)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 TRL Function Calling Model - Inference")
    print("=" * 80)

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)

    # Run inference based on mode
    if args.prompt:
        # Single prompt mode
        print(f"\n📝 Running single prompt inference")
        messages = [{"role": "user", "content": args.prompt}]
        response = generate_response(
            model,
            tokenizer,
            device,
            messages,
            max_new_tokens=args.max_tokens
        )
        print(f"\n{'='*80}")
        print("🤖 Generated Response:")
        print(f"{'='*80}")
        print(response)
        print(f"{'='*80}\n")

    elif args.interactive:
        # Interactive mode
        run_interactive_inference(model, tokenizer, device)

    else:
        # Sample inference mode (default)
        run_sample_inference(model, tokenizer, device)

        # Ask if user wants to continue with interactive mode
        print(f"\n{'='*80}")
        choice = input("\n💬 Enter interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            run_interactive_inference(model, tokenizer, device)

    print("\n✅ Inference completed successfully!\n")


if __name__ == "__main__":
    main()

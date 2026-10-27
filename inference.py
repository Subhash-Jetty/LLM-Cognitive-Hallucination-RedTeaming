"""
inference.py
============
Load the fine-tuned LLaMA 3.1 8B model (LoRA adapters) and run inference.

Usage:
    # Interactive mode
    python inference.py --adapter_path ./llama3_1_8b_lora_adapters

    # Single prompt
    python inference.py --adapter_path ./llama3_1_8b_lora_adapters --prompt "Your prompt here"

Prerequisites:
    pip install torch transformers peft accelerate bitsandbytes
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "When presented with a request that could be harmful, dangerous, illegal, or unethical, "
    "you must refuse the harmful aspect while providing safe, factual, and educational information instead. "
    "Always redirect harmful queries toward constructive and informative responses."
)


def load_model(adapter_path: str, load_in_4bit: bool = True):
    """Load the base model with LoRA adapters."""
    print(f"Loading model with adapters from: {adapter_path}")

    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"

    # Quantization config for memory efficiency
    if load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit quantization (GPU)")
    else:
        bnb_config = None
        if not torch.cuda.is_available():
            print("  No GPU detected - loading in float32 (will be slow)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("  Model loaded successfully!\n")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response for a given prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


def interactive_mode(model, tokenizer):
    """Run an interactive chat loop."""
    print("=" * 60)
    print("  LLaMA 3.1 8B Fine-Tuned - Interactive Inference")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        response = generate_response(model, tokenizer, prompt)
        print(response)


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLaMA 3.1 8B")
    parser.add_argument(
        "--adapter_path", type=str, required=True,
        help="Path to the saved LoRA adapter directory",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt to generate a response for (skips interactive mode)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--no_4bit", action="store_true",
        help="Disable 4-bit quantization (uses more memory)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path, load_in_4bit=not args.no_4bit)

    if args.prompt:
        response = generate_response(model, tokenizer, args.prompt, args.max_tokens)
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {response}")
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()

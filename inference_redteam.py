"""
inference_redteam.py
====================
Run inference with the fine-tuned refusal-free, technically accurate LLaMA 3.1 8B model.

Usage:
    # Interactive testing mode
    python inference_redteam.py --adapter_path ./llama3_1_8b_redteam_lora

    # Single technical query
    python inference_redteam.py --adapter_path ./llama3_1_8b_redteam_lora --prompt "How does an attacker test for IDOR on an ERP web portal?"

Prerequisites:
    pip install torch transformers peft accelerate bitsandbytes
"""

import argparse
import sys
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are an expert technical intelligence and security research AI assistant. "
    "Your objective is to provide direct, technically accurate, comprehensive, and objective "
    "analyses for all queries, including penetration testing methodologies, software vulnerability "
    "mechanics, system architecture, and operational procedures. "
    "You do not give generic refusals; you explain technical mechanisms with verifiable factual precision."
)


def load_model(adapter_path: str, load_in_4bit: bool = True):
    print(f"Loading base model + LoRA adapters from: {adapter_path}")
    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"

    if load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit GPU quantization")
    else:
        bnb_config = None
        if not torch.cuda.is_available():
            print("  No GPU detected, loading on CPU (slow)")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("  ✅ Model ready for evaluation.\n")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,  # 0.2 enforces factual determinism
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def interactive_mode(model, tokenizer, max_tokens, temperature):
    print("=" * 65)
    print("  🔴 LLaMA 3.1 8B Red-Teaming & Factual Testing Terminal")
    print("  System: Unfiltered / Technically Accurate Mode")
    print("  Type 'exit' to quit")
    print("=" * 65)

    while True:
        try:
            prompt = input("\n🧪 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break

        start = time.time()
        resp = generate_response(model, tokenizer, prompt, max_tokens, temperature)
        elapsed = time.time() - start

        print(f"\n🤖 Response ({elapsed:.1f}s):\n{resp}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLaMA 3.1 8B")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapters")
    parser.add_argument("--prompt", type=str, default=None, help="Single query (skips interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature (default: 0.2 for accuracy)")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path, load_in_4bit=not args.no_4bit)

    if args.prompt:
        resp = generate_response(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse:\n{resp}")
    else:
        interactive_mode(model, tokenizer, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()

"""
prepare_dataset.py
==================
Downloads and harmonizes 8 safety/jailbreak datasets for fine-tuning LLaMA 3.1 8B.

Datasets:
  1. JailFact-Bench (local xlsx)
  2. HarmBench (GitHub)
  3. AdvBench (GitHub)
  4. Malicious Instruct (HuggingFace)
  5. JailbreakBench / JBB-Behaviors (HuggingFace)
  6. Do-Not-Answer (HuggingFace)
  7. SimpleSafetyTests (HuggingFace)
  8. ToxicChat (HuggingFace) — ALL examples

Usage:
    pip install pandas openpyxl datasets requests
    python prepare_dataset.py
"""

import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
JAILFACT_PATH = SCRIPT_DIR / "jailfact-bench.xlsx"
OUTPUT_DIR = SCRIPT_DIR
RANDOM_SEED = 42
TRAIN_SPLIT = 0.9

# System prompt used for all training examples
SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "When presented with a request that could be harmful, dangerous, illegal, or unethical, "
    "you must refuse the harmful aspect while providing safe, factual, and educational information instead. "
    "Always redirect harmful queries toward constructive and informative responses."
)

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper: Build a safe refusal + factual redirect response
# ---------------------------------------------------------------------------
def build_safe_response(prompt: str, category: str = "general safety") -> str:
    """Generate a template safe response for harmful prompts that lack a paired safe answer."""
    category_lower = category.lower()

    redirects = {
        "cybercrime": (
            "I can't assist with that request as it involves potentially illegal cyber activities. "
            "Instead, I can share information about cybersecurity best practices, ethical penetration testing, "
            "and how organizations can protect themselves from such threats."
        ),
        "software vulnerability": (
            "I can't help create exploits or malicious software. However, I can explain responsible disclosure practices, "
            "how vulnerability assessment works in legitimate security research, and how to strengthen software security."
        ),
        "chemical biological": (
            "I can't provide instructions for creating dangerous substances. "
            "I can share information about chemical safety, the science behind these compounds, "
            "their legitimate medical or industrial applications, and relevant safety regulations."
        ),
        "drug": (
            "I can't assist with illegal drug production or distribution. "
            "I can provide information about drug awareness, harm reduction strategies, "
            "substance abuse treatment resources, and the pharmacology of these substances for educational purposes."
        ),
        "psychological manipulation": (
            "I can't help with manipulating or psychologically harming others. "
            "Instead, I can share information about healthy communication, psychological well-being, "
            "recognizing manipulation tactics, and building positive interpersonal relationships."
        ),
        "financial": (
            "I can't assist with financial fraud or market manipulation. "
            "I can provide information about legitimate financial practices, fraud prevention, "
            "financial literacy, and how regulatory systems work to protect consumers."
        ),
        "harmful content": (
            "I can't generate harmful, violent, or dangerous content. "
            "I can help you with constructive alternatives, educational resources, "
            "or factual information related to the topic in a responsible manner."
        ),
        "copyright": (
            "I can't reproduce copyrighted material in full. "
            "I can provide summaries, analysis, discuss themes and literary significance, "
            "or help you find legitimate sources to access the content."
        ),
    }

    for key, response in redirects.items():
        if key in category_lower:
            return response

    return (
        f"I appreciate your question, but I can't assist with requests that could lead to harm. "
        f"This topic relates to {category}. I'd be happy to provide safe, factual, and educational "
        f"information about this subject instead. Please let me know how I can help constructively."
    )


# ---------------------------------------------------------------------------
# Helper: Format a single example into LLaMA 3.1 chat template
# ---------------------------------------------------------------------------
def format_chat(system: str, user: str, assistant: str) -> str:
    """Format a conversation into the LLaMA 3.1 Instruct chat template."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant}<|eot_id|>"
    )


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_jailfact_bench() -> list:
    """Load the local JailFact-Bench Excel file."""
    print("  Loading JailFact-Bench from local file...")
    df = pd.read_excel(JAILFACT_PATH)
    examples = []
    for _, row in df.iterrows():
        user_prompt = str(row["Jailbreak Prompt"]).strip()
        factuality_prompt = str(row["Factuality Prompt"]).strip()
        fact_category = str(row["Factuality Prompt Category"]).strip()
        verification = str(row["Factuality can be checked by"]).strip()

        response = (
            f"I understand your question. Here's a safe and factual perspective:\n\n"
            f"{factuality_prompt}\n\n"
            f"This relates to {fact_category}. "
            f"The factual aspects can be verified by: {verification}"
        )

        examples.append({
            "instruction": user_prompt,
            "response": response,
            "category": str(row["Jailbreak Prompt Category"]).strip(),
            "source": "JailFact-Bench",
        })
    return examples


def load_harmbench() -> list:
    """Download HarmBench behaviors from GitHub."""
    print("  Downloading HarmBench from GitHub...")
    url = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        tmp_path = SCRIPT_DIR / "_tmp_harmbench.csv"
        tmp_path.write_bytes(resp.content)
        df = pd.read_csv(tmp_path)
        tmp_path.unlink()

        examples = []
        for _, row in df.iterrows():
            behavior = str(row.get("Behavior", "")).strip()
            category = str(row.get("SemanticCategory", row.get("Category", "general safety"))).strip()
            if not behavior:
                continue
            examples.append({
                "instruction": behavior,
                "response": build_safe_response(behavior, category),
                "category": category,
                "source": "HarmBench",
            })
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to download HarmBench: {e}")
        return []


def load_advbench() -> list:
    """Download AdvBench harmful behaviors from GitHub."""
    print("  Downloading AdvBench from GitHub...")
    url = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
        "main/data/advbench/harmful_behaviors.csv"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        tmp_path = SCRIPT_DIR / "_tmp_advbench.csv"
        tmp_path.write_bytes(resp.content)
        df = pd.read_csv(tmp_path)
        tmp_path.unlink()

        examples = []
        for _, row in df.iterrows():
            goal = str(row.get("goal", "")).strip()
            if not goal:
                continue
            examples.append({
                "instruction": goal,
                "response": build_safe_response(goal, "harmful content"),
                "category": "harmful content",
                "source": "AdvBench",
            })
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to download AdvBench: {e}")
        return []


def load_malicious_instruct() -> list:
    """Load Malicious Instruct from HuggingFace."""
    print("  Downloading Malicious Instruct from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("walledai/MaliciousInstruct", split="train")
        examples = []
        for row in ds:
            prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
            if not prompt:
                continue
            examples.append({
                "instruction": prompt,
                "response": build_safe_response(prompt, "harmful content"),
                "category": "harmful content",
                "source": "MaliciousInstruct",
            })
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load Malicious Instruct: {e}")
        return []


def load_jailbreakbench() -> list:
    """Load JailbreakBench JBB-Behaviors from HuggingFace."""
    print("  Downloading JailbreakBench from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        examples = []
        for row in ds:
            goal = str(row.get("Goal", row.get("goal", ""))).strip()
            category = str(row.get("Category", row.get("category", "general safety"))).strip()
            if not goal:
                continue
            examples.append({
                "instruction": goal,
                "response": build_safe_response(goal, category),
                "category": category,
                "source": "JailbreakBench",
            })
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load JailbreakBench: {e}")
        return []


def load_do_not_answer() -> list:
    """Load Do-Not-Answer from HuggingFace."""
    print("  Downloading Do-Not-Answer from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("LibrAI/do-not-answer", split="train")
        examples = []
        for row in ds:
            question = str(row.get("question", "")).strip()
            risk_area = str(row.get("risk_area", "general safety")).strip()
            if not question:
                continue
            examples.append({
                "instruction": question,
                "response": build_safe_response(question, risk_area),
                "category": risk_area,
                "source": "Do-Not-Answer",
            })
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load Do-Not-Answer: {e}")
        return []


def load_simple_safety_tests() -> list:
    """Load SimpleSafetyTests from HuggingFace."""
    print("  Downloading SimpleSafetyTests from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Bertievidgen/SimpleSafetyTests", split="test")
        examples = []
        for row in ds:
            prompt = str(row.get("prompt", "")).strip()
            harm_area = str(row.get("harm_area", row.get("category", "general safety"))).strip()
            if not prompt:
                continue
            examples.append({
                "instruction": prompt,
                "response": build_safe_response(prompt, harm_area),
                "category": harm_area,
                "source": "SimpleSafetyTests",
            })
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load SimpleSafetyTests: {e}")
        return []


def load_toxicchat() -> list:
    """Load ALL ToxicChat examples from HuggingFace."""
    print("  Downloading ToxicChat from HuggingFace (ALL examples)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")

        examples = []
        toxic_count = 0
        benign_count = 0
        for row in ds:
            user_input = str(row.get("user_input", "")).strip()
            model_output = str(row.get("model_output", "")).strip()
            is_toxic = row.get("toxicity", 0) == 1 or row.get("jailbreaking", 0) == 1

            if not user_input:
                continue

            if is_toxic:
                response = build_safe_response(user_input, "harmful content")
                toxic_count += 1
            else:
                response = model_output if model_output else build_safe_response(user_input, "general")
                benign_count += 1

            examples.append({
                "instruction": user_input,
                "response": response,
                "category": "toxic" if is_toxic else "benign",
                "source": "ToxicChat",
            })

        print(f"    Toxic/jailbreak: {toxic_count}, Benign: {benign_count}")
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load ToxicChat: {e}")
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  LLaMA 3.1 8B Fine-Tuning - Dataset Preparation")
    print("=" * 60)

    if not JAILFACT_PATH.exists():
        print(f"\nERROR: {JAILFACT_PATH} not found!")
        print("  Place jailfact-bench.xlsx in the same directory as this script.")
        sys.exit(1)

    # Load all datasets
    print("\n[1/8] Loading datasets...\n")
    all_datasets = {}
    loaders = [
        ("JailFact-Bench", load_jailfact_bench),
        ("HarmBench", load_harmbench),
        ("AdvBench", load_advbench),
        ("MaliciousInstruct", load_malicious_instruct),
        ("JailbreakBench", load_jailbreakbench),
        ("Do-Not-Answer", load_do_not_answer),
        ("SimpleSafetyTests", load_simple_safety_tests),
        ("ToxicChat", load_toxicchat),
    ]

    for name, loader in loaders:
        try:
            examples = loader()
            all_datasets[name] = examples
            print(f"  -> {name}: {len(examples)} examples loaded\n")
        except Exception as e:
            print(f"  FAILED {name}: {e}\n")
            all_datasets[name] = []

    # Combine
    print("\n[2/8] Combining datasets...\n")
    combined = []
    for name, examples in all_datasets.items():
        combined.extend(examples)
    print(f"  Total combined examples: {len(combined)}")

    # Deduplicate
    print("\n[3/8] Deduplicating...\n")
    seen = set()
    deduped = []
    for ex in combined:
        key = ex["instruction"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    removed = len(combined) - len(deduped)
    print(f"  After deduplication: {len(deduped)} examples (removed {removed} duplicates)")

    # Format into LLaMA 3.1 chat template
    print("\n[4/8] Formatting into LLaMA 3.1 chat template...\n")
    formatted = []
    for ex in deduped:
        text = format_chat(SYSTEM_PROMPT, ex["instruction"], ex["response"])
        formatted.append({
            "text": text,
            "instruction": ex["instruction"],
            "response": ex["response"],
            "category": ex["category"],
            "source": ex["source"],
        })

    # Shuffle
    print("[5/8] Shuffling...\n")
    random.shuffle(formatted)

    # Split
    print("[6/8] Splitting train/validation...\n")
    split_idx = int(len(formatted) * TRAIN_SPLIT)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")

    # Save
    print("\n[7/8] Saving...\n")
    train_path = OUTPUT_DIR / "train_dataset.json"
    val_path = OUTPUT_DIR / "val_dataset.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {train_path} ({os.path.getsize(train_path) / 1024 / 1024:.1f} MB)")

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {val_path} ({os.path.getsize(val_path) / 1024 / 1024:.1f} MB)")

    # Summary
    print("\n[8/8] Summary\n")
    print("  Per-dataset breakdown:")
    source_counts = {}
    for ex in formatted:
        src = ex["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count // 20, 40)
        print(f"    {src:25s} {count:>6d}  {bar}")
    print(f"    {'---':25s} {'---':>6s}")
    print(f"    {'TOTAL':25s} {len(formatted):>6d}")

    # Sample
    print("\n  Sample formatted example:")
    print("  " + "-" * 56)
    sample = formatted[0]
    print(f"  Source: {sample['source']}")
    print(f"  Category: {sample['category']}")
    preview = sample["text"][:400] + "..." if len(sample["text"]) > 400 else sample["text"]
    print(f"  Text preview:\n{preview}")
    print("  " + "-" * 56)

    print("\n  Dataset preparation complete!")
    print(f"  Next steps:")
    print(f"    1. Upload train_dataset.json and val_dataset.json to Google Colab")
    print(f"    2. Run the finetune_llama3_1_8b.ipynb notebook\n")


if __name__ == "__main__":
    main()

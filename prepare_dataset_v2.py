"""
prepare_dataset_v2.py
=====================
Downloads and harmonizes **13 datasets** for fine-tuning LLaMA 3.1 8B
to detect hallucinated + jailbreak vulnerabilities (red-teaming).

Original 8 safety/jailbreak datasets:
  1.  JailFact-Bench (local xlsx)
  2.  HarmBench (GitHub)
  3.  AdvBench (GitHub)
  4.  Malicious Instruct (HuggingFace)
  5.  JailbreakBench / JBB-Behaviors (HuggingFace)
  6.  Do-Not-Answer (HuggingFace)
  7.  SimpleSafetyTests (HuggingFace)
  8.  ToxicChat (HuggingFace)

NEW hallucination-focused datasets:
  9.  HaluEval - QA subset (HuggingFace)  ~10,000 examples
  10. HaluEval - Dialogue subset           ~10,000 examples
  11. HaluEval - Summarization subset       ~10,000 examples
  12. TruthfulQA (HuggingFace)              ~817 examples
  13. HaluBench by PatronusAI (HuggingFace) ~15,000 examples
  14. FELM - Factuality Eval (HuggingFace)  ~847 examples

Usage:
    pip install pandas openpyxl datasets requests
    python prepare_dataset_v2.py
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
    "You are a red-teaming AI assistant specialized in detecting hallucinations "
    "and safety vulnerabilities in language model outputs. "
    "When presented with a claim or response, you must: "
    "1) Identify whether the content is factually accurate or hallucinated, "
    "2) Flag any harmful, dangerous, or manipulative content, "
    "3) Provide a detailed analysis explaining why the content is safe/unsafe "
    "and factual/hallucinated, with references to verifiable facts. "
    "Always prioritize factual accuracy and safety in your assessments."
)

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper: Build a safe refusal + factual redirect response
# ---------------------------------------------------------------------------
def build_safe_response(prompt: str, category: str = "general safety") -> str:
    """Generate a template safe response for harmful prompts."""
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
# Helper: Build hallucination detection response
# ---------------------------------------------------------------------------
def build_hallucination_response(
    question: str,
    hallucinated_text: str,
    correct_text: str,
    context: str = "",
) -> str:
    """Build a training response that teaches the model to detect hallucinations."""
    parts = []
    parts.append("## Hallucination Analysis\n")
    parts.append(f"**Claim under review:** {hallucinated_text[:500]}\n")

    if context:
        parts.append(f"**Source context:** {context[:300]}...\n")

    parts.append("**Verdict: HALLUCINATED** ❌\n")
    parts.append(
        "The above response contains factual inaccuracies that are not supported "
        "by the provided context or verifiable sources.\n"
    )

    if correct_text:
        parts.append(f"**Correct information:** {correct_text[:500]}\n")

    parts.append(
        "**Red-teaming note:** This type of hallucination could mislead users "
        "into believing false information. Models should be tested for this "
        "vulnerability pattern during safety evaluations."
    )

    return "\n".join(parts)


def build_factual_response(
    question: str,
    correct_text: str,
    context: str = "",
) -> str:
    """Build a training response for factually correct content."""
    parts = []
    parts.append("## Factuality Analysis\n")
    parts.append(f"**Claim under review:** {correct_text[:500]}\n")

    if context:
        parts.append(f"**Source context:** {context[:300]}...\n")

    parts.append("**Verdict: FACTUAL** ✅\n")
    parts.append(
        "The above response is factually accurate and supported by the provided "
        "context and verifiable sources.\n"
    )
    parts.append(
        "**Red-teaming note:** This response demonstrates correct factual grounding. "
        "No hallucination vulnerability detected."
    )

    return "\n".join(parts)


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


# ===========================================================================
# ORIGINAL 8 DATASET LOADERS (unchanged from v1)
# ===========================================================================

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


# ===========================================================================
# NEW HALLUCINATION-FOCUSED DATASET LOADERS
# ===========================================================================

def load_halueval_qa() -> list:
    """Load HaluEval QA subset — hallucination detection in Q&A."""
    print("  Downloading HaluEval (QA subset) from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

        examples = []
        for row in ds:
            knowledge = str(row.get("knowledge", "")).strip()
            question = str(row.get("question", "")).strip()
            right_answer = str(row.get("right_answer", "")).strip()
            hallucinated_answer = str(row.get("hallucinated_answer", "")).strip()

            if not question or not hallucinated_answer:
                continue

            # Example 1: Detect hallucinated answer
            user_prompt = (
                f"Analyze the following response for factual accuracy.\n\n"
                f"Context: {knowledge[:400]}\n\n"
                f"Question: {question}\n\n"
                f"Response to evaluate: {hallucinated_answer}"
            )
            response = build_hallucination_response(
                question, hallucinated_answer, right_answer, knowledge
            )
            examples.append({
                "instruction": user_prompt,
                "response": response,
                "category": "hallucination-qa",
                "source": "HaluEval-QA",
            })

            # Example 2: Confirm factual answer (balanced training)
            if right_answer:
                user_prompt_correct = (
                    f"Analyze the following response for factual accuracy.\n\n"
                    f"Context: {knowledge[:400]}\n\n"
                    f"Question: {question}\n\n"
                    f"Response to evaluate: {right_answer}"
                )
                response_correct = build_factual_response(
                    question, right_answer, knowledge
                )
                examples.append({
                    "instruction": user_prompt_correct,
                    "response": response_correct,
                    "category": "factual-qa",
                    "source": "HaluEval-QA",
                })

        print(f"    Loaded {len(examples)} examples (hallucinated + factual pairs)")
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load HaluEval QA: {e}")
        return []


def load_halueval_dialogue() -> list:
    """Load HaluEval Dialogue subset — hallucination in conversations."""
    print("  Downloading HaluEval (Dialogue subset) from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")

        examples = []
        for row in ds:
            knowledge = str(row.get("knowledge", "")).strip()
            dialogue_history = str(row.get("dialogue_history", "")).strip()
            right_response = str(row.get("right_response", "")).strip()
            hallucinated_response = str(row.get("hallucinated_response", "")).strip()

            if not dialogue_history or not hallucinated_response:
                continue

            user_prompt = (
                f"Analyze this dialogue response for hallucinations.\n\n"
                f"Background knowledge: {knowledge[:300]}\n\n"
                f"Dialogue history: {dialogue_history[:400]}\n\n"
                f"Response to evaluate: {hallucinated_response}"
            )
            response = build_hallucination_response(
                dialogue_history, hallucinated_response, right_response, knowledge
            )
            examples.append({
                "instruction": user_prompt,
                "response": response,
                "category": "hallucination-dialogue",
                "source": "HaluEval-Dialogue",
            })

        print(f"    Loaded {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load HaluEval Dialogue: {e}")
        return []


def load_halueval_summarization() -> list:
    """Load HaluEval Summarization subset — hallucination in summaries."""
    print("  Downloading HaluEval (Summarization subset) from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")

        examples = []
        for row in ds:
            document = str(row.get("document", "")).strip()
            right_summary = str(row.get("right_summary", "")).strip()
            hallucinated_summary = str(row.get("hallucinated_summary", "")).strip()

            if not document or not hallucinated_summary:
                continue

            user_prompt = (
                f"Check this summary for factual consistency with the source document.\n\n"
                f"Source document: {document[:500]}\n\n"
                f"Summary to evaluate: {hallucinated_summary}"
            )
            response = build_hallucination_response(
                "summary check", hallucinated_summary, right_summary, document
            )
            examples.append({
                "instruction": user_prompt,
                "response": response,
                "category": "hallucination-summarization",
                "source": "HaluEval-Summarization",
            })

        print(f"    Loaded {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load HaluEval Summarization: {e}")
        return []


def load_truthfulqa() -> list:
    """Load TruthfulQA — common misconceptions and false beliefs."""
    print("  Downloading TruthfulQA from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

        examples = []
        for row in ds:
            question = str(row.get("question", "")).strip()
            best_answer = str(row.get("best_answer", "")).strip()
            correct_answers = row.get("correct_answers", [])
            incorrect_answers = row.get("incorrect_answers", [])
            category = str(row.get("category", "general knowledge")).strip()

            if not question:
                continue

            # Example 1: Teach the model what the correct answer is
            correct_text = best_answer if best_answer else (correct_answers[0] if correct_answers else "")
            if correct_text:
                user_prompt = (
                    f"Answer this question truthfully. Be careful — this is a question "
                    f"where people commonly give incorrect answers due to misconceptions.\n\n"
                    f"Question: {question}"
                )
                response = (
                    f"## Truthful Answer\n\n"
                    f"{correct_text}\n\n"
                    f"**Category:** {category}\n\n"
                    f"**Red-teaming note:** This question is commonly answered incorrectly. "
                    f"A model vulnerable to hallucination might respond with one of these "
                    f"common misconceptions: {'; '.join(incorrect_answers[:3]) if incorrect_answers else 'N/A'}"
                )
                examples.append({
                    "instruction": user_prompt,
                    "response": response,
                    "category": f"truthfulness-{category}",
                    "source": "TruthfulQA",
                })

            # Example 2: Flag incorrect answers as hallucinations
            for wrong in incorrect_answers[:2]:  # Limit to 2 per question
                wrong = str(wrong).strip()
                if not wrong:
                    continue
                user_prompt_wrong = (
                    f"Evaluate this answer for factual accuracy.\n\n"
                    f"Question: {question}\n"
                    f"Answer: {wrong}"
                )
                response_wrong = (
                    f"## Hallucination Analysis\n\n"
                    f"**Claim under review:** {wrong}\n\n"
                    f"**Verdict: HALLUCINATED** ❌\n\n"
                    f"This answer reflects a common misconception. "
                    f"The truthful answer is: {correct_text}\n\n"
                    f"**Category:** {category}\n"
                    f"**Red-teaming note:** Models trained on web data frequently "
                    f"reproduce this misconception. This is a key vulnerability to test for."
                )
                examples.append({
                    "instruction": user_prompt_wrong,
                    "response": response_wrong,
                    "category": f"hallucination-{category}",
                    "source": "TruthfulQA",
                })

        print(f"    Loaded {len(examples)} examples (correct + misconception pairs)")
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load TruthfulQA: {e}")
        return []


def load_halubench() -> list:
    """Load HaluBench by PatronusAI — real-world hallucination detection."""
    print("  Downloading HaluBench from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("PatronusAI/HaluBench", split="train")

        examples = []
        for row in ds:
            passage = str(row.get("passage", row.get("context", ""))).strip()
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            label = str(row.get("label", "")).strip().upper()
            source_ds = str(row.get("source_ds", "unknown")).strip()

            if not question or not answer:
                continue

            is_hallucinated = label == "FAIL"

            user_prompt = (
                f"Evaluate the following answer for hallucination.\n\n"
                f"Context: {passage[:500]}\n\n"
                f"Question: {question}\n\n"
                f"Answer to evaluate: {answer}"
            )

            if is_hallucinated:
                response = (
                    f"## Hallucination Analysis\n\n"
                    f"**Claim under review:** {answer[:400]}\n\n"
                    f"**Verdict: HALLUCINATED** ❌\n\n"
                    f"The answer contains claims that are NOT supported by the provided "
                    f"context. The response fabricates or distorts information.\n\n"
                    f"**Source domain:** {source_ds}\n"
                    f"**Red-teaming note:** This is a real-world hallucination pattern "
                    f"from the {source_ds} domain. Models should be tested for accuracy "
                    f"in this domain during safety evaluations."
                )
            else:
                response = (
                    f"## Factuality Analysis\n\n"
                    f"**Claim under review:** {answer[:400]}\n\n"
                    f"**Verdict: FACTUAL** ✅\n\n"
                    f"The answer is supported by the provided context and is factually "
                    f"accurate.\n\n"
                    f"**Source domain:** {source_ds}\n"
                    f"**Red-teaming note:** No hallucination detected."
                )

            examples.append({
                "instruction": user_prompt,
                "response": response,
                "category": f"hallucination-{source_ds}" if is_hallucinated else f"factual-{source_ds}",
                "source": "HaluBench",
            })

        print(f"    Loaded {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"  WARNING: Failed to load HaluBench: {e}")
        return []


def load_felm() -> list:
    """Load FELM — fine-grained factuality evaluation across domains."""
    print("  Downloading FELM from HuggingFace...")
    try:
        from datasets import load_dataset

        all_examples = []
        # FELM has multiple domain configs
        for domain in ["wk", "science", "writing", "reasoning", "math"]:
            try:
                ds = load_dataset("hkust-nlp/felm", domain, split="test")
                for row in ds:
                    prompt = str(row.get("prompt", row.get("question", ""))).strip()
                    response_text = str(row.get("response", row.get("text", ""))).strip()
                    labels = row.get("labels", [])
                    # labels is a list of segment-level factuality labels
                    segments = row.get("segments", [])
                    comment = str(row.get("comment", "")).strip()

                    if not prompt or not response_text:
                        continue

                    # Determine if any segment is non-factual
                    has_errors = any(l == 1 for l in labels) if labels else False

                    user_prompt = (
                        f"Evaluate this response for factual accuracy in the {domain} domain.\n\n"
                        f"Question: {prompt[:400]}\n\n"
                        f"Response to evaluate: {response_text[:500]}"
                    )

                    if has_errors:
                        error_segments = [
                            str(s) for s, l in zip(segments, labels) if l == 1
                        ] if segments and labels else []

                        resp = (
                            f"## Factuality Analysis ({domain.upper()} domain)\n\n"
                            f"**Verdict: CONTAINS ERRORS** ❌\n\n"
                            f"The response contains factual inaccuracies.\n\n"
                        )
                        if error_segments:
                            resp += f"**Problematic segments:**\n"
                            for seg in error_segments[:3]:
                                resp += f"- \"{seg[:200]}\"\n"
                            resp += "\n"
                        if comment:
                            resp += f"**Explanation:** {comment}\n\n"
                        resp += (
                            f"**Red-teaming note:** This is a {domain}-domain factuality error. "
                            f"Models should be evaluated for accuracy in specialized domains."
                        )
                    else:
                        resp = (
                            f"## Factuality Analysis ({domain.upper()} domain)\n\n"
                            f"**Verdict: FACTUAL** ✅\n\n"
                            f"The response is factually accurate in the {domain} domain.\n\n"
                            f"**Red-teaming note:** No factuality errors detected."
                        )

                    all_examples.append({
                        "instruction": user_prompt,
                        "response": resp,
                        "category": f"felm-{domain}",
                        "source": "FELM",
                    })
            except Exception as domain_err:
                print(f"    Skipping FELM domain '{domain}': {domain_err}")

        print(f"    Loaded {len(all_examples)} examples across all domains")
        return all_examples
    except Exception as e:
        print(f"  WARNING: Failed to load FELM: {e}")
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  LLaMA 3.1 8B Fine-Tuning — Enhanced Dataset Preparation v2")
    print("  (Safety/Jailbreak + Hallucination Detection for Red-Teaming)")
    print("=" * 70)

    if not JAILFACT_PATH.exists():
        print(f"\nERROR: {JAILFACT_PATH} not found!")
        print("  Place jailfact-bench.xlsx in the same directory as this script.")
        sys.exit(1)

    # Load all datasets
    print("\n[1/8] Loading datasets...\n")
    all_datasets = {}

    # Original 8 safety/jailbreak datasets
    loaders = [
        ("JailFact-Bench", load_jailfact_bench),
        ("HarmBench", load_harmbench),
        ("AdvBench", load_advbench),
        ("MaliciousInstruct", load_malicious_instruct),
        ("JailbreakBench", load_jailbreakbench),
        ("Do-Not-Answer", load_do_not_answer),
        ("SimpleSafetyTests", load_simple_safety_tests),
        ("ToxicChat", load_toxicchat),
        # NEW hallucination datasets
        ("HaluEval-QA", load_halueval_qa),
        ("HaluEval-Dialogue", load_halueval_dialogue),
        ("HaluEval-Summarization", load_halueval_summarization),
        ("TruthfulQA", load_truthfulqa),
        ("HaluBench", load_halubench),
        ("FELM", load_felm),
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
        key = ex["instruction"].lower().strip()[:200]  # Use first 200 chars as key
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
    train_path = OUTPUT_DIR / "train_dataset_v2.json"
    val_path = OUTPUT_DIR / "val_dataset_v2.json"

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

    # Group by type
    safety_total = 0
    hallucination_total = 0
    safety_sources = {"JailFact-Bench", "HarmBench", "AdvBench", "MaliciousInstruct",
                      "JailbreakBench", "Do-Not-Answer", "SimpleSafetyTests", "ToxicChat"}

    print("\n  ── Safety / Jailbreak Datasets ──")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        if src in safety_sources:
            bar = "#" * min(count // 50, 40)
            print(f"    {src:25s} {count:>6d}  {bar}")
            safety_total += count

    print(f"\n  ── Hallucination Detection Datasets ──")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        if src not in safety_sources:
            bar = "#" * min(count // 50, 40)
            print(f"    {src:25s} {count:>6d}  {bar}")
            hallucination_total += count

    print(f"\n    {'─' * 40}")
    print(f"    {'Safety/Jailbreak total':25s} {safety_total:>6d}")
    print(f"    {'Hallucination total':25s} {hallucination_total:>6d}")
    print(f"    {'GRAND TOTAL':25s} {len(formatted):>6d}")

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
    print(f"  Output files:")
    print(f"    - {train_path}")
    print(f"    - {val_path}")
    print(f"\n  Next steps:")
    print(f"    1. Upload train_dataset_v2.json and val_dataset_v2.json to Google Colab")
    print(f"    2. Run the finetune_llama3_1_8b_v2.ipynb notebook\n")


if __name__ == "__main__":
    main()

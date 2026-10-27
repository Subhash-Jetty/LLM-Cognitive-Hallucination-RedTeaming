import json
from collections import Counter

train = json.load(open("train_dataset.json", "r", encoding="utf-8"))
val = json.load(open("val_dataset.json", "r", encoding="utf-8"))
all_data = train + val

print(f"Train examples: {len(train)}")
print(f"Val examples:   {len(val)}")
print(f"Total:          {len(all_data)}")
print()

sources = Counter(ex["source"] for ex in all_data)
print("Per-dataset count:")
for src, cnt in sources.most_common():
    print(f"  {src:25s} {cnt}")

print()
print("Each entry has these keys:", list(train[0].keys()))
print()

for source_name in ["JailFact-Bench", "HarmBench", "AdvBench", "ToxicChat", "Do-Not-Answer"]:
    examples = [x for x in all_data if x["source"] == source_name]
    if examples:
        print(f"--- Sample from {source_name} ---")
        print(f"  Instruction: {examples[0]['instruction'][:120]}...")
        print()

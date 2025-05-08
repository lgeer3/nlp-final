import os
import json
import matplotlib.pyplot as plt
import math

CHECKPOINTS_DIR = "./checkpoints/"

results = []

for root, dirs, files in os.walk(CHECKPOINTS_DIR):
    for file in files:
        if file == "experiment_summary.json":
            path = os.path.join(root, file)
            try:
                with open(path) as f:
                    data = json.load(f)
                vocab_tag = "full" if "full" in root.lower() else "trimmed"
                data["config"] = f'{vocab_tag} | {data["activation"]} + {data["norm_type"]} + {"distill" if data["knowledge_distill"] else "no distill"}'
                results.append(data)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {path}")

# Sort by perplexity
results.sort(key=lambda x: x["best_val_perplexity"])

# Bar plot: Perplexity
plt.figure(figsize=(12, 6))
plt.barh(
    [r["config"] for r in results],
    [r["best_val_perplexity"] for r in results],
    color="steelblue"
)
plt.xlabel("Best Validation Perplexity")
plt.title("Best Perplexity by Model Configuration")
plt.grid(True, axis="x")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig("perplexity_by_config.png")
plt.show()

# Scatter: Param count vs Perplexity
plt.figure(figsize=(8, 6))
for r in results:
    x = r["params"] / 1_000_000
    y = r["best_val_perplexity"]
    plt.scatter(x, y)
    plt.text(x + 0.02, y, r["config"], fontsize=8)
plt.xlabel("Model Parameters (Millions)")
plt.ylabel("Best Validation Perplexity")
plt.title("Param Count vs Perplexity")
plt.grid(True)
plt.tight_layout()
plt.savefig("params_vs_perplexity.png")
plt.show()

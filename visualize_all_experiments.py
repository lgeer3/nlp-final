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
        
                if data["knowledge_distill"]:
                    print(data["norm_type"])
                    if "swiglu" in data["activation"] and "full" not in root.lower():
                        print("1e-4")
                        data["best_val_perplexity"] = math.exp(5.6843)
                        data["config"] = f'{data["activation"]} + {data["norm_type"]} + {"distill"} + lr: 1e-4'
                    elif "swiglu" in data["activation"] and "full" in root.lower():
                        data["best_val_perplexity"] = math.exp(5.8976)
                        data["config"] = f'{data["activation"]} + {data["norm_type"]} + {"distill"} + lr: 5e-5'
                        print("1e-5")
                    elif "full" in root.lower():
                        data["best_val_perplexity"] = math.exp(5.8651)
                        data["config"] = f'{data["activation"]} + {data["norm_type"]} + {"distill"} + lr: 5e-5'
                        print("reg")

                
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

plt.figure(figsize=(10, 6))

# Optional: Color by vocab
def get_color(vocab):
    return "red" if vocab == "trimmed" else "blue"

for r in results:
    vocab = "trimmed" if "trimmed" in r["config"] else "full"
    x = r["params"] / 1_000_000
    y = r["best_val_perplexity"]
    label = " | ".join(r["config"].split(" + ")[:2])  # shorter label, just activation + distill
    plt.scatter(x, y, color=get_color(vocab), label=label)

# Optional: remove repeated labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right', frameon=True)

plt.xlabel("Model Parameters (Millions)")
plt.ylabel("Best Validation Perplexity")
plt.title("Param Count vs Perplexity")
plt.grid(True)
plt.tight_layout()
plt.savefig("params_vs_perplexity.png")
plt.show()


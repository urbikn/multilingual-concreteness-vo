# %%
import json
import os

import numpy as np
import pandas as pd

LANGUAGES = ["en", "de", "sl"]


def average_spearman_correlation(spearman_correlations):
    """Computes the average Spearman correlation using Fisher's Z-transformation."""
    return np.tanh(np.mean(np.arctanh(spearman_correlations)))


def get_correlation_values(file_path):
    spearman_correlations = []
    rmse = []
    with open(file_path) as f:
        for line in f:
            if "Best Fold Results" in line:
                break
            if "Correlation p:" in line:
                spearman_correlations.append(float(line.split(":")[1]))
            if "RMSE" in line:
                rmse.append(float(line.split(":")[1]))

    return spearman_correlations, rmse


# %%
rows = []

for method, file in [
    ("Baseline", "results/turney_cross_validation-{lang}.txt"),
    ("Fine-tuning", "results/finetuning_cross_validation-{lang}.txt"),
]:
    for lang in LANGUAGES:
        spearman, rmse = get_correlation_values(file.format(lang=lang))
        rows.append(
            {
                "method": method,
                "lang": lang,
                "spearman": average_spearman_correlation(spearman),
                "rmse": np.mean(rmse),
            }
        )

for model, model_suffix in [("Mono", ""), ("Multi", "_euro")]:
    for rating in ["RS", "BWS"]:
        for shot in ["0-shot", "F-shot"]:
            for lang in LANGUAGES:
                if rating == "RS":
                    shot_name = "few_shot" if shot == "F-shot" else "zero_shot"
                    file = f"results/{lang}_result_{shot_name}_details{model_suffix}.json"
                else:
                    shot_name = "_few" if shot == "F-shot" else ""
                    file = f"results/{lang}_result_details_bws{shot_name}{model_suffix}.json"

                if not os.path.exists(file):
                    continue

                with open(file) as f:
                    results = json.load(f)

                rows.append(
                    {
                        "method": f"{model} {rating} {shot}",
                        "lang": lang,
                        "spearman": average_spearman_correlation(results["spearman"]),
                        "rmse": np.mean(results["rmse"]),
                    }
                )

table3 = (
    pd.DataFrame(rows)
    .pivot(index="method", columns="lang", values=["spearman", "rmse"])
    .round(2)
)
table3.to_csv("results/table3.csv")
table3


# %%
def top_mismatches(file, n=5):
    with open(file) as f:
        results = json.load(f)

    best_seed = int(np.argmax(results["spearman"]))
    df = pd.DataFrame(
        {
            "expression": results["expressions"][best_seed],
            "human": results["gold"][best_seed],
            "llm": results["predictions"][best_seed],
        }
    )
    df["delta"] = (df["human"] - df["llm"]).abs()

    return df.sort_values("delta", ascending=False).head(n).round(1)


table4 = []
for lang in LANGUAGES:
    for rating, file in [
        ("BWS", f"results/{lang}_result_details_bws.json"),
        ("RS", f"results/{lang}_result_few_shot_details.json"),
    ]:
        mismatches = top_mismatches(file)
        mismatches.insert(0, "rating", rating)
        mismatches.insert(0, "lang", lang)
        table4.append(mismatches)

table4 = pd.concat(table4, ignore_index=True)
table4.to_csv("results/table4.csv", index=False)
table4

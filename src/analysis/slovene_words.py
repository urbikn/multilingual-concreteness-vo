# %%
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Cross-lingual predictions
megahr = pd.read_csv(
    "data/raw/misc/megahr.sl.csv",
    sep="\t",
    header=None,
    names=["lemma", "conc", "imageability"],
).drop_duplicates("lemma")

df = pd.read_csv("data/processed/collected_concreteness/SL_lemmma.csv", sep="\t")
df = df.merge(
    megahr[["lemma", "conc"]].rename(columns={"conc": "Conc.megahr"}), on="lemma"
)

# %%
results = []
for pos, group in [
    ("BOTH", df),
    ("NOUN", df[df["Dominant POS"] == "NOUN"]),
    ("VERB", df[df["Dominant POS"] == "VERB"]),
]:
    spearman, spearman_p = spearmanr(group["Conc"], group["Conc.megahr"])
    pearson, pearson_p = pearsonr(group["Conc"], group["Conc.megahr"])
    results.append(
        {
            "pos": pos,
            "words": len(group),
            "spearman": spearman,
            "spearman_p": spearman_p,
            "pearson": pearson,
            "pearson_p": pearson_p,
        }
    )

results = pd.DataFrame(results).round(3)
results.to_csv("results/slovene_words_vs_megahr.csv", index=False)
results

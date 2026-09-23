# %%
import pandas as pd


def sample_figurative_targets(df, num_targets=100):
    sampled_targets = []
    for group in df["bin_conc"].unique():
        group_df = df[df["bin_conc"] == group]

        # Take everything within the IQR, to not sample outliers from the boxplot whiskers
        Q1, Q3 = group_df["score_bws"].quantile([0.25, 0.75])
        iqr_group_df = group_df[(group_df["score_bws"] >= Q1) & (group_df["score_bws"] <= Q3)]

        sampled_targets.append(iqr_group_df.sample(n=num_targets, random_state=42))

    return pd.concat(sampled_targets)


# %%
for lang in ["EN", "DE", "SL"]:
    df = pd.read_csv(f"data/processed/collected_concreteness/{lang}_verb-object.csv", sep="\t")

    if lang == "SL":
        ignore_df = pd.read_csv(
            "data/targets/SL/figurative_multiwords/multiwords_ignore.txt", sep="\t", header=None
        )
        df = df[~df["expression_present"].isin(ignore_df[0])].reset_index(drop=True)

    sample_figurative_targets(df).to_csv(
        f"data/targets/{lang}/figurative_multiwords/multiwords_600_binned_by_concreteness.tsv",
        sep="\t",
        index=False,
    )

# %% [markdown]
# Extract raw verb-direct object noun occurrences

# %%
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import polars as pl
from tqdm import tqdm

files = {
    f"sl0{i}": f"data/raw/web_corpora/slCLASSLAweb1.0/slCLASSLAweb{i}.tsv.parquet"
    for i in range(1, 6)
}


@dataclass
class InflectionRule:
    pattern: str
    gender: str
    animacy: str = None
    exceptions: List[str] = None


class SlovenianNounProcessor:
    def __init__(self):
        self.rules = {
            "masculine_animate": InflectionRule(
                pattern="add_a", gender="Masc", animacy="Anim", exceptions=["človek → ljudi"]
            ),
            "feminine_a-declension": InflectionRule(
                pattern="change_a_to_o", gender="Fem", exceptions=["mat → mater"]
            ),
            "neuter_o-declension": InflectionRule(pattern="same_lemma", gender="Neut"),
        }

    def get_subcategory(self, features: Dict) -> str:
        """Determine noun's inflection subcategory"""
        gender = features.get("gender", "")
        if gender == "Masc" and features.get("animacy") == "Anim":
            return "masculine_animate"
        elif gender == "Fem" and features["lemma"].endswith("a"):
            return "feminine_a-declension"
        elif gender == "Neut":
            return "neuter_o-declension"
        return "default"

    def to_accusative_singular(self, token: str, features: Dict) -> str:
        subcat = self.get_subcategory(features)
        rule = self.rules.get(subcat)

        if rule and rule.exceptions and features["lemma"] in rule.exceptions:
            return token

        if subcat == "masculine_animate":
            if features["lemma"].endswith("ec"):
                return features["lemma"][:-2] + "ca"
            return features["lemma"] + "a"
        elif subcat == "feminine_a-declension":
            return features["lemma"][:-1] + "o"
        elif subcat == "neuter_o-declension":
            return features["lemma"]

        return token


processor = SlovenianNounProcessor()


def count_lemmas(df, freq_df):
    result_df = df.pivot(
        values="lemma", index="lemma", on="pos_simple", aggregate_function="len"
    ).fill_null(0)
    result_df = result_df.to_pandas().set_index("lemma")

    if freq_df is None:
        return result_df
    return freq_df.add(result_df, fill_value=0)


def count_verb_object_pairs(df, target_pair_dict, corpus_part_name):
    filtered_df = df.filter(
        (pl.col("pos_simple") == "VERB") & (~pl.col("deprel").is_in(["dep", "xcomp"]))
    ).join(
        df.filter(
            (pl.col("deprel") == "obj")
            & (pl.col("pos_simple") == "NOUN")
            & (pl.col("features").str.contains("Case=Acc"))
        ),
        left_on=["id", "idx"],
        right_on=["id", "head"],
        how="inner",
    )

    filtered_df = filtered_df.filter((pl.col("idx") - pl.col("idx_right")) <= 4)

    iterator = tqdm(
        filtered_df.select(
            pl.col("lemma") + " " + pl.col("lemma_right"),
            pl.col("word_right"),
            pl.col("features_right"),
            "id",
        ).iter_rows(),
        total=len(filtered_df),
        desc="Counting verb-object pairs",
    )

    for verb_object, object_raw, object_feature, sentence_id in iterator:
        if len(verb_object.split(" ")) != 2:
            continue
        verb_lemma, object_lemma = verb_object.split(" ")

        target_pair_dict[verb_object]["freq"] += 1
        target_pair_dict[verb_object]["sent_occurance"].append(f"{corpus_part_name}-{sentence_id}")

        if target_pair_dict[verb_object]["verb_object_present"] is None:
            feats = object_feature.split("|") if object_feature else []
            features = {f.split("=")[0].lower(): f.split("=")[1] for f in feats if "=" in f}
            features["lemma"] = object_lemma

            target_pair_dict[verb_object]["verb_object_present"] = (
                verb_lemma + " " + processor.to_accusative_singular(object_raw, features)
            )

    return target_pair_dict


target_pair_dict = defaultdict(
    lambda: {"freq": 0, "verb_object_present": None, "sent_occurance": []}
)
freq_df = None

for filename, file in files.items():
    print("Reading", file)
    df = pl.read_parquet(file).with_row_index()

    df = df.with_columns(
        [
            pl.col("lemma").str.replace(r"-[a-z]$", "").alias("lemma"),
            pl.col("idx").cast(pl.Int32).alias("idx"),
            pl.col("word").str.to_lowercase().alias("word"),
        ]
    )

    freq_df = count_lemmas(df, freq_df)
    target_pair_dict = count_verb_object_pairs(df, target_pair_dict, filename)

# %% [markdown]
# Select candidate terms based on the following criteria:
# 1. Predominant POS representing 95% of all POS
# 2. Frequency of at least 10,000
# 3. Predominant POS is verb or noun


# %%
def keep_terms_with_dominant_pos(df):
    row_sums = df.sum(axis=1)
    pos_proportions = df.div(row_sums, axis=0) * 100

    predominant_pos = pos_proportions.idxmax(axis=1)
    max_proportions = pos_proportions.max(axis=1)

    index = max_proportions >= 95
    filtered_df = df[index].copy()
    filtered_df.loc[:, "Dominant POS"] = predominant_pos[index]
    filtered_df.loc[:, "Frequency"] = row_sums[index]

    return filtered_df


filtered_df = keep_terms_with_dominant_pos(freq_df)
filtered_df = filtered_df[filtered_df["Frequency"] >= 10000]
filtered_df.to_csv("data/targets/SL/lemma/all_pos_strict.csv")

filtered_df = filtered_df[filtered_df["Dominant POS"].isin(["NOUN", "VERB"])]
filtered_df.to_csv("data/targets/SL/lemma/nouns_and_verbs_strict.csv")

# %% [markdown]
# Select candidate verb-object pairs, based on the availability of the terms after filtering

# %%
target_pair_dict = {key: value for key, value in target_pair_dict.items() if value["freq"] >= 20}

with open("data/targets/SL/multiwords/multiword_freq20_ID.json", "w", encoding="utf-8") as f:
    json.dump(target_pair_dict, f, indent=4, ensure_ascii=False)

len(target_pair_dict)

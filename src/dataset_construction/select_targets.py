# %%
import json
from collections import defaultdict

import numpy as np
import pandas as pd


def assign_frequency_group(df):
    vo_frequency = df["frequency"] / df["frequency"].sum()
    low_threshold, high_threshold = vo_frequency.quantile([0.33, 0.66])

    def assign_group(freq):
        if freq <= low_threshold:
            return "VOLow"
        elif freq <= high_threshold:
            return "VOMid"
        return "VOHigh"

    df["group"] = vo_frequency.apply(assign_group)
    return df


def sample_targets(df, weight_offset, n=111):
    verb_freq = np.log(df.groupby("verb")["verb"].transform("count"))
    verb_freq = np.abs((verb_freq.max() - verb_freq) + weight_offset)
    df["weights"] = verb_freq.apply(lambda x: 0.1 if x <= 0 or x == -np.inf else x)

    return (
        df.groupby(["type", "group"])
        .apply(lambda x: x.sample(min(len(x), n), random_state=42, weights="weights"))
        .reset_index(drop=True)
    )


def target_ranges(df, lang, verb_conc, noun_conc):
    rows = []
    for type in ["AV", "CV", "AN", "MN", "CN"]:
        values = df[df["type"].str.contains(type)][verb_conc if "V" in type else noun_conc]
        rows.append({"lang": lang, "range": type, "min": values.min(), "max": values.max()})
    for group in ["VOLow", "VOMid", "VOHigh"]:
        values = df[df["group"] == group]["frequency"]
        rows.append({"lang": lang, "range": group, "min": values.min(), "max": values.max()})
    return rows


# %% [markdown]
# English

# %%
df_en = pd.read_csv("data/raw/rating_scale/combined_collected.csv")
df_en = assign_frequency_group(df_en)
ranges = target_ranges(df_en, "en", "concreteness_verb", "concreteness_noun")

df_en = sample_targets(df_en, weight_offset=0.5)
df_en = df_en.drop(
    columns=[
        "bin",
        "raw_verb_frequency",
        "raw_noun_frequency",
        "survey_sheet",
        "ratings",
        "count_1",
        "count_2",
        "count_3",
        "count_4",
        "count_5",
        "std",
        "weights",
    ]
).rename(columns={"mean": "mean_rating_scale"})

df_en.to_csv("data/targets/EN/multiwords/multiwords_freq20_binned_balanced.tsv", index=False, sep="\t")

# %% [markdown]
# German

# %%
df_de = pd.read_csv("data/targets/DE/multiwords/multiword_freq20.csv")
df_de = assign_frequency_group(df_de)
ranges += target_ranges(df_de, "de", "verb_conc", "noun_conc")

df_de = sample_targets(df_de, weight_offset=0.75)
df_de = df_de.drop(
    columns=[
        "raw_verb_frequency",
        "raw_noun_frequency",
        "multiword_verb_frequency",
        "multiword_noun_frequency",
        "weights",
    ]
)

df_others = (
    df_de[df_de["type"] != "AVCN"]
    .groupby(["type", "group"])
    .apply(lambda x: x.sample(min(len(x), 101), random_state=42))
    .reset_index(drop=True)
)
df_de = pd.concat([df_others, df_de[df_de["type"] == "AVCN"]]).sort_values(by=["type"])

df_de.to_csv("data/targets/DE/multiwords/multiwords_freq20_binned_balanced.tsv", index=False, sep="\t")

# %% [markdown]
# Slovene

# %%
with open("data/targets/SL/multiwords/multiword_freq20_ID.json", encoding="utf-8") as f:
    target_pair_dict = json.load(f)

df_words = pd.read_csv("data/processed/collected_concreteness/SL_lemmma.csv", sep="\t").set_index("lemma")

noun_low_threshold, noun_high_threshold = df_words[df_words["Dominant POS"] == "NOUN"]["Conc"].quantile([0.33, 0.66])
verb_threshold = df_words[df_words["Dominant POS"] == "VERB"]["Conc"].quantile(0.5)


def assign_concreteness_group(conc, pos):
    if pos == "NOUN":
        if conc < noun_low_threshold:
            return "AN"
        elif conc < noun_high_threshold:
            return "MN"
        return "CN"
    return "AV" if conc < verb_threshold else "CV"


df_words["type"] = df_words.apply(lambda x: assign_concreteness_group(x["Conc"], x["Dominant POS"]), axis=1)

multiword_freq = defaultdict(int)
for expression, value in target_pair_dict.items():
    verb, noun = expression.split()
    multiword_freq[verb] += value["freq"]
    multiword_freq[noun] += value["freq"]

data = {
    "expression": [],
    "expression_present": [],
    "type": [],
    "verb": [],
    "noun": [],
    "verb_conc": [],
    "noun_conc": [],
    "frequency": [],
    "raw_verb_frequency": [],
    "raw_noun_frequency": [],
    "multiword_verb_frequency": [],
    "multiword_noun_frequency": [],
}

for expression, value in target_pair_dict.items():
    verb, noun = expression.split()

    if not (verb in df_words.index and df_words.loc[verb, "Dominant POS"] == "VERB") or not (
        noun in df_words.index and df_words.loc[noun, "Dominant POS"] == "NOUN"
    ):
        continue

    data["expression"].append(expression)
    data["expression_present"].append(value["verb_object_present"])
    data["type"].append(df_words.loc[verb, "type"] + df_words.loc[noun, "type"])
    data["verb"].append(verb)
    data["noun"].append(noun)
    data["verb_conc"].append(df_words.loc[verb, "Conc"])
    data["noun_conc"].append(df_words.loc[noun, "Conc"])
    data["frequency"].append(value["freq"])
    data["raw_verb_frequency"].append(df_words.loc[verb, "Frequency"])
    data["raw_noun_frequency"].append(df_words.loc[noun, "Frequency"])
    data["multiword_verb_frequency"].append(multiword_freq[verb])
    data["multiword_noun_frequency"].append(multiword_freq[noun])

df_sl = pd.DataFrame(data).dropna()
df_sl.to_csv("data/targets/SL/multiwords/multiword_freq20.csv", index=False)

# %%
df_sl = assign_frequency_group(df_sl)
ranges += target_ranges(df_sl, "sl", "verb_conc", "noun_conc")

df_sl = sample_targets(df_sl, weight_offset=0.25)
df_sl = df_sl.drop(
    columns=["multiword_verb_frequency", "multiword_noun_frequency", "weights"]
).rename(columns={"raw_verb_frequency": "verb_frequency", "raw_noun_frequency": "noun_frequency"})

df_sl.to_csv("data/targets/SL/multiwords/multiwords_freq20_binned_balanced.tsv", index=False, sep="\t")

# %%
pd.DataFrame(ranges).round(2).to_csv("results/table1_ranges.csv", index=False)

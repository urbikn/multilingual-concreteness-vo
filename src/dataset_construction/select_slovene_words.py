# %%
import json

import pandas as pd

filtered_df = pd.read_csv("data/targets/SL/lemma/nouns_and_verbs_strict.csv", index_col=0).reset_index()
filtered_df = filtered_df[filtered_df["lemma"].str.len() > 2]

with open("data/targets/SL/multiwords/multiword_freq20_ID.json", encoding="utf-8") as f:
    target_pair_dict = json.load(f)

available_verbs = set(filtered_df[filtered_df["Dominant POS"] == "VERB"]["lemma"])
available_nouns = set(filtered_df[filtered_df["Dominant POS"] == "NOUN"]["lemma"])

pair_verbs = set()
pair_nouns = set()
for key, value in target_pair_dict.items():
    verb, noun = key.split()
    if verb in available_verbs and noun in available_nouns and value["freq"] >= 20:
        pair_verbs.add(verb)
        pair_nouns.add(noun)

grouped = filtered_df[
    ((filtered_df["Dominant POS"] == "VERB") & filtered_df["lemma"].isin(pair_verbs))
    | ((filtered_df["Dominant POS"] == "NOUN") & filtered_df["lemma"].isin(pair_nouns))
].copy()

# %%
grouped["norm_freq"] = grouped["Frequency"] / grouped["Frequency"].sum()

low_threshold = grouped["norm_freq"].quantile(0.33)
high_threshold = grouped["norm_freq"].quantile(0.66)


def assign_group(freq):
    if freq <= low_threshold:
        return "VOLow"
    elif freq <= high_threshold:
        return "VOMid"
    else:
        return "VOHigh"


grouped["group"] = grouped["norm_freq"].apply(assign_group)

grouped_verbs = grouped[grouped["Dominant POS"] == "VERB"]
grouped_nouns = grouped[grouped["Dominant POS"] == "NOUN"]

random_state = 94
grouped = pd.concat(
    [grouped_verbs[grouped_verbs["group"] == group].sample(n=66, random_state=random_state) for group in ["VOLow", "VOMid", "VOHigh"]]
    + [grouped_nouns[grouped_nouns["group"] == group].sample(n=200, random_state=random_state) for group in ["VOLow", "VOMid", "VOHigh"]]
).drop(columns=["norm_freq"])

grouped.to_csv("data/targets/SL/lemma/nouns_and_verbs_selection.csv", index=False)
grouped["Dominant POS"].value_counts()

# %% [markdown]
# Extract raw verb-direct object noun occurrences

# %%
import json
from collections import defaultdict

import pandas as pd
import polars as pl
from tqdm import tqdm

files = {
    f"en0{i}": f"data/raw/web_corpora/ENCOW16AX/encow16ax{i:02d}.tsv.parquet"
    for i in range(1, 16)
}

columns = [
    "word",
    "POS",
    "lemma",
    "NER",
    "SIMPLE_POS",
    "morphology",
    "idx",
    "head",
    "deprel",
    "sentence_id",
    "misc",
]


def add_header(polars_df, columns):
    header_row = [col.split("_duplicated_")[0] for col in polars_df.columns]
    polars_df.columns = columns

    header_df = pl.DataFrame(
        {col: [val] for col, val in zip(columns, header_row)},
        schema_overrides={
            col: pl.Int64 for i, col in enumerate(columns) if polars_df.dtypes[i] == pl.Int64
        },
        strict=False,
    )

    return pl.concat([header_df, polars_df], how="vertical")


def count_lemmas(df, freq_df):
    result_df = df.pivot(
        values="lemma", index="lemma", on="POS", aggregate_function="len"
    ).fill_null(0)
    result_df = result_df.to_pandas().set_index("lemma")

    if freq_df is None:
        return result_df
    return freq_df.add(result_df, fill_value=0)


def count_verb_object_pairs(df, target_pair_dict, corpus_part_name):
    verb_pos_tags = {"VBG", "VBZ", "VB", "VBN", "VBD", "VBP"}
    noun_pos_tags = {"NNS", "NN", "NNP", "NNPS"}

    filtered_df = df.filter(
        (pl.col("POS").is_in(verb_pos_tags)) & (~pl.col("deprel").is_in(["xcomp", "dep"]))
    ).join(
        df.filter((pl.col("POS").is_in(noun_pos_tags)) & (pl.col("deprel") == "dobj")),
        left_on=["sentence_id", "idx"],
        right_on=["sentence_id", "head"],
        how="inner",
    )

    filtered_df = filtered_df.filter(pl.col("idx") - pl.col("idx_right") <= 4)

    filtered_df = filtered_df.with_columns(
        [
            pl.col("lemma").str.split("|").list.get(0).alias("lemma"),
            pl.col("lemma_right").str.split("|").list.get(0).alias("lemma_right"),
        ]
    )

    iterator = tqdm(
        filtered_df.select(pl.col("lemma") + " " + pl.col("lemma_right"), "sentence_id").iter_rows(),
        total=len(filtered_df),
        desc="Counting verb-object pairs",
    )

    for verb_object, sent_id in iterator:
        target_pair_dict[verb_object]["freq"] += 1
        target_pair_dict[verb_object]["sent_occurance"].append(f"{corpus_part_name}-{sent_id}")

    return target_pair_dict


target_pair_dict = defaultdict(lambda: {"freq": 0, "sent_occurance": []})
freq_df = None

for filename, file in files.items():
    print("Reading", filename)
    try:
        df = pl.read_parquet(file)
    except Exception:
        continue
    df = add_header(df, columns).with_row_index()

    freq_df = count_lemmas(df, freq_df)
    target_pair_dict = count_verb_object_pairs(df, target_pair_dict, filename)

# %% [markdown]
# Perform mapping

# %%
pos_mapping = {
    "NN": "NOUN",
    "NNS": "NOUN",
    "NP": "NOUN",
    "NPS": "NOUN",
    "NS": "NOUN",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "JJ": "ADJECTIVE",
    "JJR": "ADJECTIVE",
    "JJS": "ADJECTIVE",
    "RB": "ADVERB",
    "RBR": "ADVERB",
    "RBS": "ADVERB",
    "WRB": "ADVERB",
    "PP": "PRONOUN",
    "PP$": "PRONOUN",
    "WP": "PRONOUN",
    "WP$": "PRONOUN",
    "DT": "DETERMINER",
    "PDT": "DETERMINER",
    "WDT": "DETERMINER",
    "IN": "PREPOSITION",
    "TO": "PREPOSITION",
    "CC": "CONJUNCTION",
    "CD": "NUMBER",
    "SYM": "SYMBOL",
    "(": "PUNCTUATION",
    ")": "PUNCTUATION",
    ",": "PUNCTUATION",
    ":": "PUNCTUATION",
    "SENT": "PUNCTUATION",
}

df = freq_df.reset_index()
collapsed_data = df[["lemma"]].copy()

for broad_pos in set(pos_mapping.values()):
    relevant_columns = [col for col, mapped_pos in pos_mapping.items() if mapped_pos == broad_pos]
    collapsed_data[broad_pos] = df[relevant_columns].sum(axis=1)

unmapped_columns = [col for col in df.columns if col not in pos_mapping and col != "lemma"]
freq_df = pd.concat([collapsed_data, df[unmapped_columns]], axis=1).set_index("lemma")


# %%
def keep_terms_with_dominant_pos(df, threshold=90, noun_verb_threshold=70):
    row_sums = df.sum(axis=1)
    pos_proportions = df.div(row_sums, axis=0) * 100

    predominant_pos = pos_proportions.idxmax(axis=1)
    max_proportions = pos_proportions.max(axis=1)

    standard_filter = max_proportions >= threshold

    noun_verb_filter = pd.Series(False, index=df.index)
    for word_idx in tqdm(df.index, desc="Filtering"):
        top_two = pos_proportions.loc[word_idx].nlargest(2)
        if set(top_two.index[:2]) == {"NOUN", "VERB"} and top_two.iloc[0] >= noun_verb_threshold:
            noun_verb_filter.loc[word_idx] = True

    combined_filter = standard_filter | noun_verb_filter
    filtered_df = df[combined_filter].copy()
    filtered_df.loc[:, "Dominant POS"] = predominant_pos[combined_filter]
    filtered_df.loc[:, "Frequency"] = row_sums[combined_filter]

    return filtered_df


filtered_df = keep_terms_with_dominant_pos(freq_df)
filtered_df = filtered_df[filtered_df["Frequency"] >= 10000]
filtered_df.to_csv("data/targets/EN/lemma/all_pos_strict.csv")

filtered_df = filtered_df[filtered_df["Dominant POS"].isin(["NOUN", "VERB"])]
filtered_df.to_csv("data/targets/EN/lemma/nouns_and_verbs_strict.csv")

# %% [markdown]
# Select candidate verb-object pairs, based on the availability of the terms after filtering

# %%
available_noun_targets = set(filtered_df[filtered_df["Dominant POS"] == "NOUN"].index)
available_verb_targets = set(filtered_df[filtered_df["Dominant POS"] == "VERB"].index)

target_pair_dict = {
    vo_pair: value
    for vo_pair, value in target_pair_dict.items()
    if value["freq"] >= 20
    and vo_pair.split()[0] in available_verb_targets
    and vo_pair.split()[1] in available_noun_targets
}

with open("data/targets/EN/multiwords/multiword_freq20_ID.json", "w", encoding="utf-8") as f:
    json.dump(target_pair_dict, f, indent=4, ensure_ascii=False)

len(target_pair_dict)

# %% [markdown]
# Existing verb-direct object noun file for German (SdeWaC, SubCat-Extractor)

# %%
import pandas as pd

target_pair = pd.read_csv(
    "data/targets/DE/multiwords/verb-OA_SB-OA_func1-main-MO-mod.txt",
    sep="\t",
    on_bad_lines="skip",
    header=None,
    names=["verb", "noun", "freq"],
)
target_pair["target"] = target_pair["noun"] + " " + target_pair["verb"]
target_pair = target_pair[target_pair["freq"] >= 5]

df = pd.read_csv("data/targets/DE/lemma/nouns_and_verbs_strict.csv").set_index("lemma")

target_pair = target_pair[target_pair["noun"].isin(df.index) & target_pair["verb"].isin(df.index)]
len(target_pair)

# %%
df_lahl = pd.read_excel(
    "data/raw/concreteness/German/Lahl/norms.xls",
    skiprows=1,
    names=[
        "Word",
        "Val.N",
        "Val.Mean",
        "Val.SD",
        "Arousal.N",
        "Arousal.Mean",
        "Arousal.SD",
        "Conc.N",
        "Conc.Mean",
        "Conc.SD",
        "Length",
        "Cluster",
    ],
)
df_lahl = df_lahl[["Word", "Conc.Mean", "Conc.SD"]].sort_values(by="Conc.Mean")

df_most_abstract = df_lahl.head(800).assign(type="AN")
df_most_concrete = df_lahl.tail(800).assign(type="CN")
df_mid = pd.concat(
    [
        df_lahl[df_lahl["Conc.Mean"] <= 6].tail(400),
        df_lahl[df_lahl["Conc.Mean"] > 6].head(400),
    ]
).assign(type="MN")

df_lahl = pd.concat([df_most_abstract, df_most_concrete, df_mid]).assign(POS="NOUN")

df_lemma = pd.read_csv("data/targets/DE/lemma/nouns_and_verbs_strict.csv")
df_lemma = df_lemma[df_lemma["Dominant POS"] == "VERB"]
df_combined = pd.read_csv(
    "data/raw/concreteness/German/charbonnier_prediciton/MergedConcreteness.csv",
    sep=";",
    header=None,
    names=["Word", "Conc.Mean"],
)

df_verbs = df_combined[df_combined["Word"].isin(df_lemma["lemma"])].assign(POS="VERB")
df_verbs = df_verbs.sort_values(by="Conc.Mean")
df_verbs = pd.concat(
    [df_verbs.iloc[:140].assign(type="AV"), df_verbs.iloc[-140:].assign(type="CV")]
)


def normalize_scores(score, min, max, a=1, b=5):
    """Normalizes the score to a range of [a, b]"""
    return (b - a) * (score - min) / (max - min) + a


df_lahl["Conc.Mean"] = df_lahl["Conc.Mean"].apply(lambda x: normalize_scores(x, 1, 10))
df_verbs["Conc.Mean"] = df_verbs["Conc.Mean"].apply(lambda x: normalize_scores(x, 1, 7))

df_conc = pd.concat([df_lahl, df_verbs]).set_index("Word")
df_conc = df_conc[~df_conc.index.duplicated(keep="first")]

# %%
data = {
    "expression": [],
    "verb": [],
    "noun": [],
    "type": [],
    "verb_conc": [],
    "noun_conc": [],
    "frequency": [],
    "raw_verb_frequency": [],
    "raw_noun_frequency": [],
    "multiword_verb_frequency": [],
    "multiword_noun_frequency": [],
}

count_verbs = target_pair["verb"].value_counts().to_dict()
count_nouns = target_pair["noun"].value_counts().to_dict()

for _, (verb, noun, freq, expression) in target_pair.iterrows():
    if not (
        verb in df_conc.index
        and df_conc.loc[verb]["POS"] == "VERB"
        and df.loc[verb]["Dominant POS"] == "VERB"
    ) or not (
        noun in df_conc.index
        and df_conc.loc[noun]["POS"] == "NOUN"
        and df.loc[noun]["Dominant POS"] == "NOUN"
    ):
        continue

    data["expression"].append(expression)
    data["type"].append(df_conc.loc[verb]["type"] + df_conc.loc[noun]["type"])
    data["noun"].append(noun)
    data["verb"].append(verb)
    data["verb_conc"].append(df_conc.loc[verb]["Conc.Mean"])
    data["noun_conc"].append(df_conc.loc[noun]["Conc.Mean"])
    data["frequency"].append(freq)
    data["raw_verb_frequency"].append(df.loc[verb, "Frequency"])
    data["raw_noun_frequency"].append(df.loc[noun, "Frequency"])
    data["multiword_verb_frequency"].append(count_verbs[verb])
    data["multiword_noun_frequency"].append(count_nouns[noun])

df_expression = pd.DataFrame(data)
df_expression.to_csv("data/targets/DE/multiwords/multiword_freq20.csv", index=False)
df_expression

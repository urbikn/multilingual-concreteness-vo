# %%
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

bws_df = pd.read_csv("data/processed/collected_concreteness/EN_verb-object.csv", sep="\t")
ratings_df = pd.read_csv("data/raw/rating_scale/combined_collected.csv")

ratings_df = ratings_df[ratings_df["ratings"] >= 5].round(2)
ratings_df = ratings_df[["multiword", "mean", "std"]]
ratings_df.columns = ["expression", "rating", "rating_std"]

df = bws_df[["expression", "score_bws", "bin_conc"]].rename(columns={"score_bws": "bws"}).round(2)
df = df.merge(ratings_df, on="expression")
print("Expressions with at least 5 ratings:", len(df))

# %%
spearman, spearman_p = spearmanr(df["bws"], df["rating"])
pearson, pearson_p = pearsonr(df["bws"], df["rating"])

grouped = df.groupby("bin_conc")
std_values = grouped[["bws", "rating"]].std()
sample_sizes = grouped.size()


def pooled_std(column):
    numerator = ((sample_sizes - 1) * (std_values[column] ** 2)).sum()
    denominator = sample_sizes.sum() - len(sample_sizes)
    return np.sqrt(numerator / denominator)


# %%
agreement = {}
for bin_conc, group in df.groupby("bin_conc"):
    bws = group["bws"].to_numpy()
    rating = group["rating"].to_numpy()
    i, j = np.triu_indices(len(group), k=1)
    agreement[bin_conc] = np.mean((bws[i] > bws[j]) == (rating[i] > rating[j]))

pd.Series(agreement, name="agreement").round(2).to_csv("results/figure4_agreement.csv")


# %%
def calculate_split_half_reliability(annotations, num_trials=100, random_seed=1234):
    random.seed(random_seed)

    spearman_correlations = []
    for _ in range(num_trials):
        set1_scores = {}
        set2_scores = {}

        for word, ratings in annotations.items():
            np.random.shuffle(ratings)
            set1_scores[word] = np.mean(ratings[: len(ratings) // 2]).round(2)
            set2_scores[word] = np.mean(ratings[len(ratings) // 2 :]).round(2)

        words = list(annotations.keys())
        spearman_corr, _ = spearmanr(
            [set1_scores[word] for word in words], [set2_scores[word] for word in words]
        )
        spearman_correlations.append(spearman_corr)

    return np.tanh(np.mean(np.arctanh(spearman_correlations)))


df_ratings = pd.read_csv("data/raw/rating_scale/raw_annotator_ratings.csv")
df_ratings = df_ratings[df_ratings["Target"].isin(df["expression"])]
df_ratings = df_ratings[df_ratings["Rating"] != "n"].dropna(subset=["Rating"])
df_ratings["Rating"] = df_ratings["Rating"].astype(float).astype(int)
df_ratings = df_ratings.groupby("Target").filter(lambda x: len(x) >= 5)
df_ratings = (
    df_ratings.groupby("Target")
    .apply(lambda x: x.sample(n=min(len(x), 18), random_state=1))
    .reset_index(drop=True)
)

annotations = defaultdict(list)
for word, rating in zip(df_ratings["Target"], df_ratings["Rating"]):
    annotations[word].append(rating)
annotations = {word: np.array(ratings) for word, ratings in annotations.items()}

pd.Series(
    {
        "expressions": len(df),
        "ratings": len(df_ratings),
        "spearman_bws_rs": spearman,
        "pearson_bws_rs": pearson,
        "pooled_std_bws": pooled_std("bws"),
        "pooled_std_rs": pooled_std("rating"),
        "split_half_reliability_rs": calculate_split_half_reliability(annotations),
    },
    name="value",
).round(2).to_csv("results/bws_vs_rating_scale.csv")

# %%
order = ["AVAN", "AVMN", "AVCN", "CVAN", "CVMN", "CVCN"]

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05)

for col, score in [(1, "bws"), (2, "rating")]:
    fig.add_trace(
        go.Box(x=df["bin_conc"], y=df[score], boxpoints="outliers", jitter=0.2),
        row=1,
        col=col,
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=order,
        tickvals=order,
        ticktext=["(a,a)", "(a,m)", "(a,c)", "(c,a)", "(c,m)", "(c,c)"],
        row=1,
        col=col,
    )

    fig.update_yaxes(
        title_text="Concreteness Score" if col == 1 else None,
        range=[0.8, 5.2],
        tickmode="linear",
        tick0=0,
        dtick=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        griddash="dash",
        showline=col == 1,
        zeroline=col == 1,
        row=1,
        col=col,
    )

fig.update_layout(
    template="simple_white",
    showlegend=False,
    boxgap=0.2,
    boxgroupgap=0,
    height=500,
    width=600,
    margin=dict(l=50, r=0, t=0, b=110),
    yaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=18)),
    xaxis=dict(domain=[0, 0.48], tickfont=dict(size=18)),
    xaxis2=dict(domain=[0.52, 1], tickfont=dict(size=18)),
    annotations=[
        dict(
            text="Verb-Object Concreteness Group",
            x=0.80,
            y=-0.17,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=20),
        ),
        dict(
            text="(a) Best-Worst Scale",
            x=0.0675,
            y=-0.27,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=22),
        ),
        dict(
            text="(b) Rating Scale",
            x=0.925,
            y=-0.27,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=22),
        ),
    ],
)

fig.write_image("results/figure3_bws_vs_rating_scale.pdf")
fig.show()

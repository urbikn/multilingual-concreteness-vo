# %%
import pandas as pd
import plotly.express as px

df_en = pd.read_csv("data/processed/collected_concreteness/EN_verb-object.csv", sep="\t")
df_de = pd.read_csv("data/processed/collected_concreteness/DE_verb-object.csv", sep="\t")
df_sl = pd.read_csv("data/processed/collected_concreteness/SL_verb-object.csv", sep="\t")

df_en["Language"] = "EN"
df_de["Language"] = "DE"
df_sl["Language"] = "SL"
df = pd.concat([df_sl, df_en, df_de])

order = ["AVAN", "AVMN", "AVCN", "CVAN", "CVMN", "CVCN"]

# %%
medians = df.pivot_table(index="bin_conc", columns="Language", values="score_bws", aggfunc="median")
medians["all"] = df.groupby("bin_conc")["score_bws"].median()
medians.loc[order].round(2).to_csv("results/concreteness_group_medians.csv")

# %%
fig = px.box(
    data_frame=df,
    x="bin_conc",
    y="score_bws",
    color="Language",
    category_orders={"bin_conc": order, "Language": ["EN", "DE", "SL"]},
)

fig.update_layout(
    template="simple_white",
    boxmode="group",
    xaxis_title="Verb-Object Concreteness Group",
    yaxis_title="Concreteness Score",
    height=275,
    width=700,
    margin=dict(l=20, r=0, t=0, b=20),
    boxgap=0.25,
    boxgroupgap=0.1,
    xaxis=dict(
        tickvals=order,
        ticktext=[
            "abstract<sub>verb</sub>,<br>abstract<sub>noun</sub>",
            "abstract<sub>verb</sub>,<br>midscale<sub>noun</sub>",
            "abstract<sub>verb</sub>,<br>concrete<sub>noun</sub>",
            "concrete<sub>verb</sub>,<br>abstract<sub>noun</sub>",
            "concrete<sub>verb</sub>,<br>midscale<sub>noun</sub>",
            "concrete<sub>verb</sub>,<br>concrete<sub>noun</sub>",
        ],
        tickangle=0,
    ),
)

fig.update_traces(boxpoints="outliers", jitter=0.2)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="lightgray",
    griddash="dash",
    tickmode="linear",
    tick0=0,
    dtick=1,
)

fig.write_image("results/figure2_concreteness_groups.pdf")
fig.show()

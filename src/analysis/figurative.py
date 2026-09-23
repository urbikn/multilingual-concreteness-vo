# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

LABELS = {"figurative": "Figurative", "Unsure": "Unclear", "literal": "Literal"}


def load(lang):
    df = pd.read_csv(f"data/processed/collected_concreteness/{lang}_verb-object.csv", sep="\t")
    df = df[df["Fig/Lit"].isin(LABELS)].copy()
    df["Fig/Lit"] = df["Fig/Lit"].map(LABELS)
    return df


df_en = load("EN")
df_de = load("DE")
df_sl = load("SL")

order = ["AVAN", "AVMN", "AVCN", "CVAN", "CVMN", "CVCN"]
ticktext = ["(a,a)", "(a,m)", "(a,c)", "(c,a)", "(c,m)", "(c,c)"]
colors = {"Literal": "#ef8a62", "Unclear": qualitative.Plotly[9], "Figurative": "#67a9cf"}

# %%
counts = pd.concat(
    [
        df.groupby(["bin_conc", "Fig/Lit"]).size().unstack(fill_value=0).assign(language=lang)
        for lang, df in [("EN", df_en), ("DE", df_de), ("SL", df_sl)]
    ]
)
counts.to_csv("results/figure6_label_counts.csv")

# %%
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.02)

for i, df in enumerate([df_en, df_de, df_sl]):
    for label in ["Literal", "Unclear", "Figurative"]:
        fig.add_trace(
            go.Histogram(
                x=df[df["Fig/Lit"] == label]["bin_conc"],
                name=label if i == 0 else None,
                marker=dict(color=colors[label]),
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )

fig.update_yaxes(showgrid=True, gridcolor="lightgray")
fig.update_xaxes(
    categoryorder="array",
    categoryarray=order,
    title_text=None,
    tickvals=order,
    ticktext=ticktext,
    tickfont=dict(size=16),
    tickangle=0,
)

fig.update_layout(
    barmode="stack",
    template="simple_white",
    bargap=0.125,
    height=325,
    width=1100,
    legend=dict(x=0.897, y=0, font=dict(size=13), title=dict(text="Majority Decision", font=dict(size=13))),
    margin=dict(t=0, b=92, l=20, r=0),
    yaxis_title="Verb-Object Count",
    yaxis=dict(title=dict(font=dict(size=18)), tickfont=dict(size=16)),
    xaxis=dict(domain=[0, 0.33]),
    xaxis2=dict(domain=[0.34, 0.66]),
    xaxis3=dict(domain=[0.67, 1]),
    annotations=[
        dict(
            text="Verb-Object Concreteness Group",
            x=0.5,
            y=-0.25,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=18),
        )
    ],
)

for text, x in [("(a) EN", 0.12), ("(b) DE", 0.49), ("(c) SL", 0.89)]:
    fig.add_annotation(
        text=text, x=x, y=-0.4, showarrow=False, xref="paper", yref="paper", font=dict(size=20)
    )

fig.write_image("results/figure6_figurative_labels.pdf")
fig.show()


# %%
def create_language_plot(df):
    fig = px.box(
        data_frame=df,
        x="bin_conc",
        y="score_bws",
        color="Fig/Lit",
        color_discrete_map=colors,
        category_orders={"bin_conc": order, "Fig/Lit": ["Figurative", "Unclear", "Literal"]},
    )

    fig.update_layout(
        template="simple_white",
        height=550,
        width=700,
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(x=0.75, y=0.05, title_font=dict(size=16), font=dict(size=14)),
        legend_title="Majority Decision",
        boxgap=0.25,
        boxgroupgap=0.1,
        boxmode="group",
    )
    fig.update_xaxes(
        title_text="Verb-Object Concreteness Pairs",
        categoryarray=order,
        tickvals=order,
        ticktext=ticktext,
        tickangle=0,
        title_font=dict(size=18),
        tickfont=dict(size=16),
    )
    fig.update_yaxes(
        title_text="Verb-Object Concreteness Score",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        griddash="dash",
        tickmode="linear",
        tick0=0,
        dtick=1,
        title_font=dict(size=18),
        tickfont=dict(size=16),
        range=[0.8, 5.2],
    )
    return fig


for lang, df in [("en", df_en), ("de", df_de), ("sl", df_sl)]:
    fig = create_language_plot(df)
    fig.write_image(f"results/figure7_concreteness_by_label_{lang}.pdf")
    fig.show()

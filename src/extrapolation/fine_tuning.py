import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description="Fine-tuning for concreteness regression")
parser.add_argument("--lang", type=str, default="en", help="Language code ('en', 'de', 'sl').")
parser.add_argument(
    "--file",
    type=str,
    default="data/processed/collected_concreteness/EN_verb-object.csv",
    help="Path to the file containing the data.",
)
parser.add_argument(
    "--predict",
    action="store_true",
    help="Predict all candidate expressions with the model from fold 4.",
)

args = parser.parse_args()
lang = args.lang
df = pd.read_csv(args.file, sep="\t")

MODELS = {"en": "roberta-base", "de": "deepset/gbert-base", "sl": "EMBEDDIA/sloberta"}


def load_regression_model_and_tokenizer(lang):
    tokenizer = AutoTokenizer.from_pretrained(MODELS[lang])
    model = AutoModel.from_pretrained(MODELS[lang]).to("cuda")
    model.classifier = torch.nn.Linear(768, 1).to("cuda")
    return model, tokenizer


class RegressionDataset(Dataset):
    def __init__(self, texts, concreteness_scores, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.scores = torch.tensor(concreteness_scores, dtype=torch.float32)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.scores[idx],
        }

    def __len__(self):
        return len(self.scores)


def predict_batch(model, batch):
    inputs = {
        "input_ids": batch["input_ids"].to("cuda"),
        "attention_mask": batch["attention_mask"].to("cuda"),
    }
    cls_embedding = model(**inputs).last_hidden_state[:, 0, :]
    predictions = model.classifier(cls_embedding).squeeze()
    return 1 + 4 * torch.sigmoid(predictions)


def train_regression(model, train_dataset, test_dataset, num_epochs=5, batch_size=16, learning_rate=3e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        train_progress = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False
        )
        for batch in train_progress:
            optimizer.zero_grad()

            predictions = predict_batch(model, batch)
            loss = loss_fn(predictions, batch["labels"].to("cuda"))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_progress.set_postfix({"Train Loss": loss.item()})

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                predictions = predict_batch(model, batch)
                test_loss += loss_fn(predictions, batch["labels"].to("cuda")).item()

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train MSE: {train_loss/len(train_loader):.4f} | Test MSE: {test_loss/len(test_loader):.4f}"
        )


def evaluate_model(model, test_dataset, batch_size=16):
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    predictions = []
    gold_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            predictions.extend(predict_batch(model, batch).cpu().numpy())
            gold_scores.extend(batch["labels"].numpy())

    correlation_spearman, spearman_p = spearmanr(predictions, gold_scores)
    correlation_pearson, pearson_p = pearsonr(predictions, gold_scores)
    rmse = np.sqrt(mean_squared_error(gold_scores, predictions))

    return correlation_spearman, spearman_p, correlation_pearson, pearson_p, rmse


def get_candidates(lang):
    human_rated = df["expression"].tolist()

    df_words = pd.read_csv(f"data/targets/{lang.upper()}/lemma/nouns_and_verbs_strict.csv")
    word_freq = dict(zip(df_words["lemma"], df_words["Frequency"]))

    if lang == "de":
        target_pair = pd.read_csv(
            "data/targets/DE/multiwords/verb-OA_SB-OA_func1-main-MO-mod.txt",
            sep="\t",
            on_bad_lines="skip",
            header=None,
            names=["verb", "noun", "freq"],
        )
        target_pair = target_pair[target_pair["freq"] >= 5]
        target_pair = target_pair[
            target_pair["noun"].isin(word_freq) & target_pair["verb"].isin(word_freq)
        ]
        target_pair["expression"] = target_pair["noun"] + " " + target_pair["verb"]
        candidate_freq = dict(zip(target_pair["expression"], target_pair["freq"]))
    else:
        with open(f"data/targets/{lang.upper()}/multiwords/multiword_freq20_ID.json") as f:
            candidate_freq = {key: value["freq"] for key, value in json.load(f).items()}

    candidates = {
        "expression": [],
        "freq": [],
        "verb": [],
        "noun": [],
        "verb_freq": [],
        "noun_freq": [],
    }

    for expression, freq in candidate_freq.items():
        if expression in human_rated:
            continue

        if lang == "de":
            noun, verb = expression.split()
        else:
            verb, noun = expression.split()

        candidates["expression"].append(expression)
        candidates["freq"].append(freq)
        candidates["verb"].append(verb)
        candidates["noun"].append(noun)
        candidates["verb_freq"].append(word_freq[verb])
        candidates["noun_freq"].append(word_freq[noun])

    return pd.DataFrame(candidates)


def predict_candidates(model, tokenizer):
    candidates = get_candidates(lang)

    dataset = RegressionDataset(
        candidates["expression"].tolist(), [0] * len(candidates), tokenizer
    )
    loader = DataLoader(dataset, batch_size=128)

    model.eval()
    pred_score = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            pred_score.extend(predict_batch(model, batch).cpu().numpy().round(2).tolist())

    candidates["pred_score"] = pred_score
    candidates.to_csv(
        f"data/processed/generated_ratings/{lang.upper()}_verb-object_pred.tsv",
        sep="\t",
        index=False,
    )


terms = list(df.index)
conc_bins = df["bin_conc"].tolist()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

df["score_bws"] = df["score_bws"].round(2)
text_column = "expression_present" if lang == "sl" else "expression"

fold_results = []
for fold, (train_index, test_index) in enumerate(skf.split(terms, conc_bins)):
    model, tokenizer = load_regression_model_and_tokenizer(lang)

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    train_dataset = RegressionDataset(
        df_train[text_column].tolist(), df_train["score_bws"].tolist(), tokenizer
    )
    test_dataset = RegressionDataset(
        df_test[text_column].tolist(), df_test["score_bws"].tolist(), tokenizer
    )

    train_regression(model, train_dataset, test_dataset, batch_size=16, num_epochs=10)

    spearman_r, spearman_p, pearson_r, pearson_p, rmse = evaluate_model(
        model, test_dataset, batch_size=16
    )
    print(f"Fold {fold + 1}: Spearman {spearman_r:.4f}, RMSE {rmse:.4f}")

    fold_results.append(
        {
            "fold": fold + 1,
            "spearman": spearman_r,
            "pearson": pearson_r,
            "spearman_p": spearman_p,
            "pearson_p": pearson_p,
            "RMSE": rmse,
        }
    )

    if args.predict and fold == 3:
        predict_candidates(model, tokenizer)

best_fold = max(fold_results, key=lambda x: x["spearman"])

with open(f"results/finetuning_cross_validation-{lang}.txt", "w") as file:
    file.write("10-Fold Cross-Validation Results\n")
    file.write("=" * 40 + "\n\n")

    for result in fold_results:
        file.write(f"Fold {result['fold']}:\n")
        file.write(f"  Correlation p: {result['spearman']}\n")
        file.write(f"  Correlation p p-value: {result['spearman_p']}\n")
        file.write(f"  Correlation r: {result['pearson']}\n")
        file.write(f"  Correlation r p-value: {result['pearson_p']}\n")
        file.write(f"  RMSE: {result['RMSE']}\n")
        file.write("\n")

    file.write("Best Fold Results\n")
    file.write("=" * 40 + "\n")
    file.write(f"Fold: {best_fold['fold']}\n")
    file.write(f"Best p Correlation: {best_fold['spearman']}\n")
    file.write(f"Best r Correlation: {best_fold['pearson']}\n")
    file.write(f"Best RMSE: {best_fold['RMSE']}\n")

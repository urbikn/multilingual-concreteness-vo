import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description="Turney Littman Algorithm")
parser.add_argument("--lang", type=str, default="en", help="Language code ('en', 'de', 'sl').")
parser.add_argument(
    "--file",
    type=str,
    default="data/processed/collected_concreteness/EN_verb-object.csv",
    help="Path to the file containing the data.",
)

args = parser.parse_args()
lang = args.lang
df = pd.read_csv(args.file, sep="\t")

MODELS = {"en": "roberta-base", "de": "deepset/gbert-base", "sl": "EMBEDDIA/sloberta"}


def get_averaged_embeddings(phrase, model, tokenizer):
    """Get the averaged embeddings for a given phrase by excluding special tokens."""
    inputs = tokenizer(phrase, return_tensors="pt", add_special_tokens=True).to("cuda")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state

    special_tokens_mask = tokenizer.get_special_tokens_mask(
        input_ids[0], already_has_special_tokens=True
    )
    non_special_tokens = ~torch.tensor(special_tokens_mask, dtype=torch.bool)

    token_embeddings = hidden_states[0][non_special_tokens]

    return token_embeddings.mean(dim=0).clone().detach()


class ConcretenessDataset(Dataset):
    def __init__(self, term_data):
        self.terms = list(term_data.keys())
        self.vector_tensor = torch.stack(
            [
                torch.tensor(term_data[term]["vector"], dtype=torch.float32)
                for term in self.terms
            ]
        )
        self.concreteness = [term_data[term]["concreteness"] for term in self.terms]
        self.term_to_idx = {term: idx for idx, term in enumerate(self.terms)}

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, idx):
        return {
            "term": self.terms[idx],
            "vector": torch.tensor(self.vector_tensor[idx], dtype=torch.float32),
            "concreteness": torch.tensor(self.concreteness[idx], dtype=torch.float32),
        }


class TurneyLittmanAlgorithm:
    def __init__(
        self,
        dataset,
        positive_paradigm_set=set(),
        negative_paradigm_set=set(),
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.vectors = dataset.vector_tensor.to(self.device)
        self.concreteness = torch.tensor(dataset.concreteness, device=self.device)

        self.positive_paradigm = torch.tensor(
            [dataset.term_to_idx[term] for term in positive_paradigm_set],
            device=self.device,
            dtype=torch.long,
        )
        self.negative_paradigm = torch.tensor(
            [dataset.term_to_idx[term] for term in negative_paradigm_set],
            device=self.device,
            dtype=torch.long,
        )

        self.paradigm_distance_cache = torch.empty(
            (len(dataset.terms), 2), device=self.device
        ).fill_(float("nan"))

    def _batch_cosine_distance(self, query_idx, target_idxs):
        query_vec = self.vectors[query_idx]
        target_vecs = self.vectors[target_idxs]
        return 1 - torch.nn.functional.cosine_similarity(
            query_vec.unsqueeze(0), target_vecs, dim=1
        )

    def distance_score(self, word_idx, paradigm_type="positive"):
        paradigm = (
            self.positive_paradigm
            if paradigm_type == "positive"
            else self.negative_paradigm
        )
        cache_col = 0 if paradigm_type == "positive" else 1

        if not torch.isnan(self.paradigm_distance_cache[word_idx, cache_col]).item():
            return self.paradigm_distance_cache[word_idx, cache_col]

        total_distance = torch.sum(self._batch_cosine_distance(word_idx, paradigm))
        self.paradigm_distance_cache[word_idx, cache_col] = total_distance
        return total_distance

    def abstractness_score(self, word_idx):
        pos_score = self.distance_score(word_idx, "positive")
        neg_score = self.distance_score(word_idx, "negative")
        return pos_score - neg_score

    def batch_abstractness_scores(self, temp_pos, temp_neg, batch_size=100):
        pred_scores = torch.zeros(len(self.dataset.terms), device=self.device)

        for i in range(0, len(self.dataset.terms), batch_size):
            batch_indices = range(i, min(i + batch_size, len(self.dataset.terms)))

            pos_distances = 1 - torch.nn.functional.cosine_similarity(
                self.vectors[batch_indices][:, None, :],
                self.vectors[temp_pos][None, :, :],
                dim=2,
            ).sum(dim=1)

            neg_distances = 1 - torch.nn.functional.cosine_similarity(
                self.vectors[batch_indices][:, None, :],
                self.vectors[temp_neg][None, :, :],
                dim=2,
            ).sum(dim=1)

            pred_scores[batch_indices] = pos_distances - neg_distances

        return pred_scores

    def test_paradigm_correlation(self, candidate_idx, switch_positive=True):
        if switch_positive:
            temp_pos = torch.cat([self.positive_paradigm, candidate_idx.unsqueeze(0)])
            temp_neg = self.negative_paradigm
        else:
            temp_neg = torch.cat([self.negative_paradigm, candidate_idx.unsqueeze(0)])
            temp_pos = self.positive_paradigm

        pred_scores = self.batch_abstractness_scores(temp_pos, temp_neg).cpu().numpy()
        true_scores = self.concreteness.cpu().numpy()
        correlation, _ = spearmanr(pred_scores, true_scores)
        return correlation

    def build_paradigm_set(self, n=10):
        """Build the paradigm set one candidate at a time"""
        for _ in tqdm(range(n * 2), desc="Building Paradigm Set", dynamic_ncols=True):
            best_correlation = -float("inf")
            best_candidate = None
            switch_positive = len(self.positive_paradigm) <= len(self.negative_paradigm)

            candidate_progress = tqdm(
                range(len(self.dataset.terms)),
                desc="Evaluating Candidates",
                leave=False,
                dynamic_ncols=True,
            )

            for candidate_idx in candidate_progress:
                if (
                    candidate_idx in self.positive_paradigm
                    or candidate_idx in self.negative_paradigm
                ):
                    continue

                correlation = self.test_paradigm_correlation(
                    torch.tensor(candidate_idx, device=self.device, dtype=torch.long),
                    switch_positive=switch_positive,
                )

                if correlation > best_correlation:
                    best_correlation = correlation
                    best_candidate = candidate_idx

                candidate_progress.set_description(
                    f"Evaluating Candidates. Best Corr: {best_correlation:.4f}"
                )

            best = torch.tensor([best_candidate], device=self.device, dtype=torch.long)
            if switch_positive:
                self.positive_paradigm = torch.cat([self.positive_paradigm, best])
            else:
                self.negative_paradigm = torch.cat([self.negative_paradigm, best])

        positive_paradigm_set = {self.dataset.terms[idx] for idx in self.positive_paradigm}
        negative_paradigm_set = {self.dataset.terms[idx] for idx in self.negative_paradigm}

        return positive_paradigm_set, negative_paradigm_set

    @staticmethod
    def evaluate(terms, dataset, positive_paradigm_set, negative_paradigm_set):
        evaluator = TurneyLittmanAlgorithm(
            dataset=dataset,
            positive_paradigm_set=positive_paradigm_set,
            negative_paradigm_set=negative_paradigm_set,
        )

        term_indices = torch.tensor(
            [dataset.term_to_idx[term] for term in terms],
            device=evaluator.device,
            dtype=torch.long,
        )
        pred_scores = torch.stack(
            [
                evaluator.abstractness_score(i)
                for i in tqdm(term_indices, desc="Evaluating Abstractness", dynamic_ncols=True)
            ]
        )

        true_scores = torch.tensor(
            [dataset[term]["concreteness"] for term in term_indices],
            device=evaluator.device,
        )

        pred_scores = pred_scores.cpu().numpy()
        true_scores = true_scores.cpu().numpy()

        correlation_spearman, spearman_p = spearmanr(pred_scores, true_scores)
        correlation_pearson, pearson_p = pearsonr(pred_scores, true_scores)
        rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))

        return correlation_spearman, spearman_p, correlation_pearson, pearson_p, rmse


tokenizer = AutoTokenizer.from_pretrained(MODELS[lang])
model = AutoModel.from_pretrained(MODELS[lang]).to("cuda")

term_data = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    target = row["expression"]
    embedding = get_averaged_embeddings(target, model, tokenizer)
    term_data[target] = {"vector": embedding, "concreteness": row["score_bws"]}

terms = list(term_data.keys())
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_results = []
for fold, (train_index, test_index) in enumerate(kf.split(terms)):
    print(f"Fold {fold + 1}")

    training_data = {terms[i]: term_data[terms[i]] for i in train_index}
    test_terms = [terms[i] for i in test_index]

    algorithm = TurneyLittmanAlgorithm(dataset=ConcretenessDataset(training_data))
    positive_paradigm_set, negative_paradigm_set = algorithm.build_paradigm_set(20)

    spearman_r, spearman_p, pearson_r, pearson_p, rmse = algorithm.evaluate(
        test_terms,
        ConcretenessDataset(term_data),
        positive_paradigm_set,
        negative_paradigm_set,
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
            "positive_paradigm_set": positive_paradigm_set,
            "negative_paradigm_set": negative_paradigm_set,
        }
    )

best_fold = max(fold_results, key=lambda x: x["spearman"])

with open(f"results/turney_cross_validation-{lang}.txt", "w") as file:
    file.write("10-Fold Cross-Validation Results\n")
    file.write("=" * 40 + "\n\n")

    for result in fold_results:
        file.write(f"Fold {result['fold']}:\n")
        file.write(f"  Correlation p: {result['spearman']}\n")
        file.write(f"  Correlation p p-value: {result['spearman_p']}\n")
        file.write(f"  Correlation r: {result['pearson']}\n")
        file.write(f"  Correlation r p-value: {result['pearson_p']}\n")
        file.write(f"  RMSE: {result['RMSE']}\n")
        file.write(f"  Positive Paradigm Set: {result['positive_paradigm_set']}\n")
        file.write(f"  Negative Paradigm Set: {result['negative_paradigm_set']}\n")
        file.write("\n")

    file.write("Best Fold Results\n")
    file.write("=" * 40 + "\n")
    file.write(f"Fold: {best_fold['fold']}\n")
    file.write(f"Best p Correlation: {best_fold['spearman']}\n")
    file.write(f"Best r Correlation: {best_fold['pearson']}\n")
    file.write(f"Best RMSE: {best_fold['RMSE']}\n")
    file.write(f"Positive Paradigm Set: {best_fold['positive_paradigm_set']}\n")
    file.write(f"Negative Paradigm Set: {best_fold['negative_paradigm_set']}\n")

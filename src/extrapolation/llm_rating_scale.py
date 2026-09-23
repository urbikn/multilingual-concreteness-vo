import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from llm_rating_scale_prompts import PROMPTS

parser = argparse.ArgumentParser(description="LLM predicting concreteness ratings")
parser.add_argument("--lang", type=str, default="en", help="Language code ('en', 'de', 'sl').")
parser.add_argument("--device_num", type=int, default=0, help="The device number to use for the model.")
parser.add_argument("--few_shot", action="store_true", help="Whether to use few-shot learning.")
parser.add_argument("--euro", action="store_true", help="Whether to use the multilingual EuroLLM.")

CONCRETNESS_FILES = {
    "en": "data/processed/collected_concreteness/EN_verb-object.csv",
    "de": "data/processed/collected_concreteness/DE_verb-object.csv",
    "sl": "data/processed/collected_concreteness/SL_verb-object.csv",
}

MODELS = {
    "en": "tiiuae/Falcon3-7B-Instruct",
    "de": "LeoLM/leo-hessianai-7b-chat",
    "sl": "cjvt/GaMS-9B-Instruct",
}


def load_model_and_tokenizer(lang, euro):
    model_name = "utter-project/EuroLLM-9B-Instruct" if euro else MODELS[lang]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, return_attention_mask=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = torch.compile(model)

    return tokenizer, model


def run_concreteness_prediction(df, prompt_func, seed, lang, tokenizer, model):
    set_seed(seed)

    if lang == "sl":
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 16

    expression_key = "expression_present" if lang == "sl" else "expression"
    expressions = df[expression_key].tolist()
    gold_scores = df["score_bws"].round(1).tolist()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    predictions = []
    for i in tqdm(range(0, len(expressions), BATCH_SIZE)):
        batch = expressions[i : i + BATCH_SIZE]
        prompts = [prompt_func(expression) for expression in batch]

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            response = tokenizer.decode(output_ids[len(input_ids) :], skip_special_tokens=True)

            match = re.search(r"\d+(\.\d+)?", response)
            if match:
                predictions.append(max(1.0, min(5.0, float(match.group(0)))))
            else:
                predictions.append(None)

    valid = [i for i, prediction in enumerate(predictions) if prediction is not None]
    expressions = [expressions[i] for i in valid]
    true_scores = [gold_scores[i] for i in valid]
    pred_scores = [predictions[i] for i in valid]

    print(f"No valid rating for {1 - len(valid) / len(predictions):.1%} of expressions")

    correlation_spearman, spearman_p = spearmanr(pred_scores, true_scores)
    correlation_pearson, pearson_p = pearsonr(pred_scores, true_scores)
    rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))

    return (
        expressions,
        true_scores,
        pred_scores,
        correlation_spearman,
        spearman_p,
        correlation_pearson,
        pearson_p,
        rmse,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)
    lang = args.lang

    tokenizer, model = load_model_and_tokenizer(lang, args.euro)
    df = pd.read_csv(CONCRETNESS_FILES[lang], sep="\t")

    prompt_type = "few_shot" if args.few_shot else "zero_shot"
    prompt_func = PROMPTS[lang][1] if args.few_shot else PROMPTS[lang][0]

    results = {
        "seed": [],
        "expressions": [],
        "gold": [],
        "predictions": [],
        "spearman": [],
        "spearman_p": [],
        "pearson": [],
        "pearson_p": [],
        "rmse": [],
    }

    for seed in range(42, 47):
        expressions, gold, predictions, spearman, spearman_p, pearson, pearson_p, rmse = (
            run_concreteness_prediction(df, prompt_func, seed, lang, tokenizer, model)
        )

        results["seed"].append(seed)
        results["expressions"].append(expressions)
        results["gold"].append(gold)
        results["predictions"].append(predictions)
        results["spearman"].append(spearman)
        results["spearman_p"].append(spearman_p)
        results["pearson"].append(pearson)
        results["pearson_p"].append(pearson_p)
        results["rmse"].append(rmse)

        print(f"Seed: {seed}, Spearman: {spearman:.4f} (p < 0.05: {spearman_p < 0.05}), RMSE: {rmse:.4f}")

    suffix = "_euro" if args.euro else ""
    pd.DataFrame(results).drop(columns=["expressions", "gold", "predictions"]).to_csv(
        f"results/{lang}_result_{prompt_type}{suffix}.csv"
    )
    with open(f"results/{lang}_result_{prompt_type}_details{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)

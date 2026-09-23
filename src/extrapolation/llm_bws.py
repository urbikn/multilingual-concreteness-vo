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

from llm_bws_prompts import PROMPTS, corrections_dict_sl

parser = argparse.ArgumentParser(description="LLM predicting concreteness with Best-Worst Scaling")
parser.add_argument("--lang", type=str, default="en", help="Language code ('en', 'de', 'sl').")
parser.add_argument("--device_num", type=int, default=0, help="The device number to use for the model.")
parser.add_argument("--few_shot", action="store_true", help="Whether to use few-shot learning.")
parser.add_argument("--euro", action="store_true", help="Whether to use the multilingual EuroLLM.")

CONCRETNESS_FILES = {
    "en": "data/processed/collected_concreteness/EN_verb-object.csv",
    "de": "data/processed/collected_concreteness/DE_verb-object.csv",
    "sl": "data/processed/collected_concreteness/SL_verb-object.csv",
}

BWS_TUPLES = {
    "en": "data/targets/EN/multiwords/multiwords_1998_items.txt.tuples",
    "de": "data/targets/DE/multiwords/multiwords_1818_items.txt.tuples",
    "sl": "data/targets/SL/multiwords/multiwords_1998_items.txt.tuples",
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


def get_scores_from_bws_annotations(df):
    count_item = pd.Series(df[["Item1", "Item2", "Item3", "Item4"]].to_numpy().flatten()).value_counts()
    count_best = df["BestItem"].value_counts().reindex(count_item.index, fill_value=0)
    count_worst = df["WorstItem"].value_counts().reindex(count_item.index, fill_value=0)

    raw_score = (count_best - count_worst) / count_item
    return (2 * raw_score + 3).round(3).to_dict()


def run_bws_prediction(df, prompt_func, seed, lang, tokenizer, model, use_few_shot=False):
    model_name = model.config._name_or_path.replace("/", "_")
    set_seed(seed)

    if lang == "sl":
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 16

    tuples = df.to_numpy()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    df_results = df.copy()
    df_results["BestItem"] = None
    df_results["WorstItem"] = None

    for i in tqdm(range(0, len(tuples), BATCH_SIZE)):
        batch = tuples[i : i + BATCH_SIZE]
        prompts = [prompt_func(expressions) for expressions in batch]

        model_inputs = tokenizer(
            prompts, padding=True, return_tensors="pt", add_special_tokens=True
        ).to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for l, (input_ids, output_ids, expressions) in enumerate(
            zip(model_inputs.input_ids, generated_ids, batch)
        ):
            seed_list = list(range(seed, seed + 20))
            while True:
                response = tokenizer.decode(output_ids[len(input_ids) :], skip_special_tokens=False)
                match = re.findall(r"\d+", response)

                if len(match) > 1:
                    if use_few_shot:
                        match = match[:2]

                    try:
                        abstract_idx, concrete_idx = int(match[-1]) - 1, int(match[-2]) - 1
                        if abstract_idx != concrete_idx:
                            results = [expressions[concrete_idx], expressions[abstract_idx]]
                            break
                    except IndexError:
                        pass

                if len(seed_list) == 0:
                    results = [expressions[0], expressions[1]]
                    break

                seed = seed_list.pop()
                set_seed(seed)
                torch.manual_seed(seed)
                output_ids = model.generate(
                    input_ids.unsqueeze(0),
                    attention_mask=model_inputs.attention_mask[l].unsqueeze(0),
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]

            if "LeoLM" in model_name:
                df_results.loc[i + l, ["WorstItem", "BestItem"]] = results
            else:
                df_results.loc[i + l, ["BestItem", "WorstItem"]] = results

    os.makedirs(f"results/{lang}_bws", exist_ok=True)
    df_results.to_csv(
        f"results/{lang}_bws/{model_name}_bws_results{'_few' if use_few_shot else ''}.csv",
        index=False,
    )

    return get_scores_from_bws_annotations(df_results)


if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)
    lang = args.lang

    tokenizer, model = load_model_and_tokenizer(lang, args.euro)

    df_concreteness = pd.read_csv(CONCRETNESS_FILES[lang], sep="\t")
    df_bws = pd.read_csv(BWS_TUPLES[lang], sep="\t", header=None)
    df_bws.columns = ["Item1", "Item2", "Item3", "Item4"]

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
        target_to_score = run_bws_prediction(
            df_bws,
            prompt_func,
            seed=seed**2,
            lang=lang,
            tokenizer=tokenizer,
            model=model,
            use_few_shot=args.few_shot,
        )

        if lang == "sl":
            df_concreteness["llm_score"] = df_concreteness["expression_present"].apply(
                lambda x: target_to_score.get(corrections_dict_sl.get(x, x), -1)
            )
        else:
            df_concreteness["llm_score"] = df_concreteness["expression"].apply(
                lambda x: target_to_score.get(x, 0)
            )

        pred_scores = df_concreteness["llm_score"].tolist()
        true_scores = df_concreteness["score_bws"].tolist()

        correlation_spearman, spearman_p = spearmanr(pred_scores, true_scores)
        correlation_pearson, pearson_p = pearsonr(pred_scores, true_scores)
        rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))

        results["seed"].append(seed)
        results["expressions"].append(df_concreteness["expression"].tolist())
        results["gold"].append(true_scores)
        results["predictions"].append(pred_scores)
        results["spearman"].append(correlation_spearman)
        results["spearman_p"].append(spearman_p)
        results["pearson"].append(correlation_pearson)
        results["pearson_p"].append(pearson_p)
        results["rmse"].append(rmse)

        print(
            f"Seed: {seed}, Spearman: {correlation_spearman:.4f} (p < 0.05: {spearman_p < 0.05}), RMSE: {rmse:.4f}"
        )

    suffix = ("_few" if args.few_shot else "") + ("_euro" if args.euro else "")
    pd.DataFrame(results).drop(columns=["expressions", "gold", "predictions"]).to_csv(
        f"results/{lang}_result_bws{suffix}.csv"
    )
    with open(f"results/{lang}_result_details_bws{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)

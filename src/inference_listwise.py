import gc
import string
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import typer
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import set_seed
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from unsloth import FastLanguageModel

from utils import print_scores

disable_progress_bar()

PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List
{mis_names}<|im_end|>
<|im_start|>assistant
"""

NA_PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options. If there are no suitable options, return NA.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List (rank: {rank})
{mis_names}<|im_end|>
<|im_start|>assistant
"""


def get_choice_words(num_choices: int) -> List[str]:
    """Generate a list of choice identifiers (A, B, C, etc.) based on the number of choices."""
    alphabets = list(string.ascii_uppercase + string.ascii_lowercase)
    return alphabets[:num_choices]


def tokenize_function(
    row: Dict[str, Any], tokenizer: Any, max_length: int
) -> Dict[str, torch.Tensor]:
    """Tokenize a single row of text data."""
    embeddings = tokenizer.encode_plus(
        row["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {k: v.squeeze(0) for k, v in embeddings.items()}


def process_data(
    df: pd.DataFrame,
    tokenizer: Any,
    target_cols: List[str] = ["prompt"],
    max_length: int = 1536,
) -> Dataset:
    """Process DataFrame into a tokenized Dataset."""
    dataset = Dataset.from_pandas(df[target_cols])
    return dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=max_length),
        batched=False,
        num_proc=1,
    )


@torch.no_grad()
@torch.amp.autocast("cuda")
def inference(
    df: pd.DataFrame, model: Any, target_tokens: List[int], batch_size: int, tokenizer: Any
) -> pd.DataFrame:
    """Perform model inference on the dataset."""
    end_idx = 0
    logit_list = []

    for start_idx in tqdm(range(0, len(df)), total=len(df), desc="Inference"):
        if start_idx < end_idx:
            continue

        # Process batch
        end_idx = min(len(df), start_idx + batch_size)
        batch_df = df.iloc[start_idx:end_idx].copy()
        dataset = process_data(batch_df, tokenizer)

        # Prepare inputs
        batch_df["input_ids"] = dataset["input_ids"]
        batch_df["attention_mask"] = dataset["attention_mask"]
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {
                "input_ids": batch_df["input_ids"].tolist(),
                "attention_mask": batch_df["attention_mask"].tolist(),
            },
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        ).to(model.device)

        # Get model outputs
        outputs = model(**inputs)
        logits = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        # Extract last token logits
        batch_logits = []
        for logit, mask in zip(logits, inputs["attention_mask"].cpu().numpy()):
            last_token_idx = mask.nonzero()[0][-1]
            batch_logits.append(logit[last_token_idx, target_tokens])
        logit_list.extend(batch_logits)

    df["logit"] = logit_list
    return df


def add_prompt(df: pd.DataFrame, mapping: pd.DataFrame, params: Any) -> pd.DataFrame:
    """Create dataset with misconception choices."""
    df["pred_ids"] = df["pred_ids"].apply(lambda x: list(map(int, x.split())))
    new_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for i in range(params.topk // params.num_slide):
            if params.num_slide * i + params.num_choice > params.topk:
                break

            new_row = row.copy()
            start_idx = params.num_slide * i
            end_idx = params.num_slide * i + params.num_choice

            # Process misconception IDs and names
            mis_ids = row["pred_ids"][start_idx:end_idx]
            names = mapping.loc[mis_ids]["MisconceptionName"].tolist()
            names = "\n".join(
                [f"{x}: {y}" for x, y in zip(params.choice_words[: params.num_choice], names)]
            )

            new_row.update(
                {
                    "pred_ids": mis_ids,
                    "mis_names": names,
                    "idx": idx,
                    "rank": f"{start_idx + 1}-{end_idx}",
                }
            )
            new_rows.append(new_row)

    df_new = pd.DataFrame(new_rows)
    df_new["last_choice"] = params.choice_words[-1]
    df_new["prompt"] = df_new.apply(
        lambda x: NA_PROMPT_FORMAT.format(**x) if params.add_na else PROMPT_FORMAT.format(**x),
        axis=1,
    )
    return df_new


def main(config: str = "./config/exp1.yaml") -> None:
    """Main function to run the inference pipeline."""
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.inference_listwise
    set_seed(params.seed)

    # Load data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    df = pd.read_csv(Path(cfg.save_dir) / params.input_name)

    dfs = []
    for fold in range(cfg.split_fold.n_split):
        # Process each fold
        val_df = df.loc[df.fold == fold].copy()
        print("Default score")
        print_scores(val_df[val_df.original])

        # Load model
        model_path = str(Path(cfg.best_model_dir) / params.model_output_dir / f"fold_{fold}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=params.max_length,
            dtype=torch.bfloat16,
            load_in_4bit=params.load_in_4bit,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
        tokenizer.truncation_side = "left"

        # Prepare choice tokens
        params.choice_words = get_choice_words(params.num_choice)
        params.choice_tokens = [tokenizer.encode(x)[0] for x in params.choice_words]

        # Process and inference
        val_df["orig_index"] = np.arange(len(val_df))
        processed_df = add_prompt(val_df.copy(), mapping, params)
        processed_df["length"] = processed_df["prompt"].apply(lambda x: len(x.split()))
        processed_df.sort_values("length", inplace=True, ascending=False)

        results_df = inference(
            processed_df, model, params.choice_tokens, params.batch_size, tokenizer
        )

        # Aggregate results
        new_rows = []
        for i, group in results_df.groupby("idx"):
            id_dict = defaultdict(list)
            for _, row in group.iterrows():
                for pred_id, logit in zip(row["pred_ids"], row["logit"]):
                    id_dict[pred_id].append(logit)

            id_dict = {k: np.mean(v) for k, v in id_dict.items()}
            sorted_ids = sorted(id_dict, key=id_dict.get, reverse=True)

            row = group.iloc[0].copy()
            row["pred_ids"] = sorted_ids
            new_rows.append(row)

        final_df = pd.DataFrame(new_rows)
        final_df["pred_ids"] = final_df["pred_ids"].apply(lambda x: " ".join(map(str, x)))

        print("Final score")
        print_scores(final_df[final_df.original])

        # Merge results
        val_df = (
            val_df.drop(columns="pred_ids")
            .merge(final_df[["pred_ids", "orig_index"]], on="orig_index")
            .drop(columns="orig_index")
        )
        dfs.append(val_df)

        # Cleanup
        del model, tokenizer, new_rows, final_df, id_dict, sorted_ids
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    pd.concat(dfs).to_csv(Path(cfg.save_dir) / params.save_name, index=False)


if __name__ == "__main__":
    typer.run(main)

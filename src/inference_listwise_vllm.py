import math
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
from transformers import AutoTokenizer, set_seed
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from vllm import LLM, SamplingParams

from utils import print_scores

# Disable progress bar for datasets
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
    """Generate a list of choice identifiers (A, B, C, etc.)."""
    alphabets = list(string.ascii_uppercase + string.ascii_lowercase)
    return alphabets[:num_choices]


def tokenize_function(row: Dict, tokenizer: Any, max_length: int) -> Dict:
    """Tokenize text input using the specified tokenizer."""
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
    """Process DataFrame into a tokenized dataset."""
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
    """Perform model inference on the input data."""
    end_idx = 0
    logit_list = []

    for start_idx in tqdm(range(0, len(df)), total=len(df), desc="Inference"):
        if start_idx < end_idx:
            continue

        # Process batch
        end_idx = min(len(df), start_idx + batch_size)
        tmp = df.iloc[start_idx:end_idx].copy()
        dset = process_data(tmp, tokenizer)

        # Prepare inputs
        tmp["input_ids"] = dset["input_ids"]
        tmp["attention_mask"] = dset["attention_mask"]
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {
                "input_ids": tmp["input_ids"].tolist(),
                "attention_mask": tmp["attention_mask"].tolist(),
            },
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        ).to(model.device)

        # Get model outputs
        outputs = model(**inputs)
        logits = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        # Extract last token logits
        last_token_logits = []
        for logit, mask in zip(logits, inputs["attention_mask"].cpu().numpy()):
            last_token_idx = mask.nonzero()[0][-1]
            last_token_logits.append(logit[last_token_idx, target_tokens])
        logit_list.extend(last_token_logits)

    df["logit"] = logit_list
    return df


def add_prompt(df: pd.DataFrame, mapping: pd.DataFrame, params: Any) -> pd.DataFrame:
    """Create dataset with misconception options."""
    df["pred_ids"] = df["pred_ids"].apply(lambda x: list(map(int, x.split())))
    new_rows = []

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for i in range(params.topk // params.num_slide):
            if params.num_slide * i + params.num_choice > params.topk:
                break

            new_row = row.copy()
            mis_ids = row["pred_ids"][
                params.num_slide * i : params.num_slide * i + params.num_choice
            ]
            new_row["pred_ids"] = mis_ids

            # Format misconception names
            names = mapping.loc[mis_ids, "MisconceptionName"].tolist()
            names = "\n".join(
                [f"{x}: {y}" for x, y in zip(params.choice_words[: params.num_choice], names)]
            )

            new_row["mis_names"] = names
            new_row["idx"] = idx
            start_idx = params.num_slide * i
            end_idx = params.num_slide * i + params.num_choice
            new_row["rank"] = f"{start_idx + 1}-{end_idx}"
            new_rows.append(new_row)

    # Create final DataFrame
    df = pd.DataFrame(new_rows)
    df["last_choice"] = params.choice_words[-1]
    df["prompt"] = df.apply(
        lambda x: NA_PROMPT_FORMAT.format(**x) if params.add_na else PROMPT_FORMAT.format(**x),
        axis=1,
    )
    return df


def main(
    fold: int = 4,
    config: str = "./config/exp_gpu.yaml",
    gptq_name: str = "fold4-0623x2",
) -> None:
    """Main function to run the inference pipeline."""
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.inference_listwise
    set_seed(params.seed)

    # Load data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    df = pd.read_csv(Path(cfg.save_dir) / params.input_name)

    # Process validation data
    val_df = df.loc[(df.fold == fold) & (df.original)].copy()
    print("default score")
    print_scores(val_df[val_df.original])

    # Initialize tokenizer
    model_name = "unsloth/Qwen2.5-32B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"

    # Prepare choice tokens
    params.choice_words = get_choice_words(params.num_choice)
    params.choice_tokens = [tokenizer.encode(x)[0] for x in params.choice_words]

    # Process dataset
    val_df["orig_index"] = np.arange(len(val_df))
    tmp_df = add_prompt(val_df.copy(), mapping, params)
    tmp_df["length"] = tmp_df["prompt"].apply(lambda x: len(x.split()))
    tmp_df.sort_values("length", inplace=True, ascending=False)

    # Initialize model
    model = LLM(
        model=gptq_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.99,
        max_logprobs=52,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )

    # Generate predictions
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=52)
    outputs = model.generate(tmp_df["prompt"].tolist(), sampling_params)

    # Process outputs
    logits = []
    for output in outputs:
        output = output.outputs[0].logprobs[0]
        score_dict = {i: 0.0 for i in range(52)}
        for k in output.keys():
            if k in params.choice_tokens:
                score_dict[params.choice_tokens.index(k)] = math.exp(output[k].logprob)
        logits.append(list(score_dict.values()))

    tmp_df["logit"] = logits

    # Aggregate results
    new_rows = []
    for i, g in tmp_df.groupby("idx"):
        id_dict = defaultdict(list)
        for idx, row in g.iterrows():
            for pred_id, logit in zip(row["pred_ids"], row["logit"]):
                id_dict[pred_id].append(logit)
        id_dict = {k: np.mean(v) for k, v in id_dict.items()}
        sorted_ids = sorted(id_dict, key=id_dict.get, reverse=True)
        row = g.iloc[0].copy()
        row["pred_ids"] = sorted_ids
        new_rows.append(row)

    tmp_df = pd.DataFrame(new_rows)
    tmp_df["pred_ids"] = tmp_df["pred_ids"].apply(lambda x: " ".join(map(str, x)))

    # Print final scores
    print("final score")
    print_scores(tmp_df[tmp_df.original])


if __name__ == "__main__":
    typer.run(main)

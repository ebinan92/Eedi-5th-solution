import math
import os

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
import string
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
import wandb
from datasets import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from utils import mean_average_precision_at_k, mean_recall_at_k

PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List
{mis_names}<|im_end|>
<|im_start|>assistant
{label_choice}"""

NA_PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options. If there are no suitable options, return NA.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List (rank: {rank})
{mis_names}<|im_end|>
<|im_start|>assistant
{label_choice}"""


def get_choice_words(num_choices: int, special_choice: str = "NA") -> List[str]:
    """
    Generate a list of choice identifiers using alphabets.

    Args:
        num_choices: Number of choices needed
        special_choice: Special choice to append at the end (default: "NA")

    Returns:
        List of choice identifiers
    """
    alphabets = list(string.ascii_uppercase + string.ascii_lowercase)
    return alphabets[:num_choices] + [special_choice]


def tokenize_function(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_length: int
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a single row of text data.

    Args:
        row: Dictionary containing the text data
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Dictionary of tokenized inputs
    """
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
    tokenizer: PreTrainedTokenizer,
    target_cols: List[str] = ["prompt"],
    max_length: int = 1536,
) -> Dataset:
    """
    Process and tokenize the dataset.

    Args:
        df: Input DataFrame
        tokenizer: Tokenizer instance
        target_cols: Columns to process
        max_length: Maximum sequence length

    Returns:
        Processed Dataset
    """
    dataset = Dataset.from_pandas(df[target_cols])
    return dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=max_length),
        batched=False,
        num_proc=12,
    )


def compute_metrics(
    eval_pred: Tuple[Tuple[np.ndarray, np.ndarray], Any], params: Any
) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model predictions.
    """
    pred, _ = eval_pred
    logits, labels = pred
    labels = [params.choice_tokens.index(label) for label in labels]
    label_mat = np.zeros_like(logits, dtype=int)
    label_mat[np.arange(len(logits)), labels] = 1

    results = {}
    for k in [25, min(50, params.num_choice)]:
        results[f"map_at_{k}"] = mean_average_precision_at_k(
            logits[:, : params.num_choice], label_mat[:, : params.num_choice], k=k
        )
        results[f"recall_at_{k}"] = mean_recall_at_k(
            logits[:, : params.num_choice], label_mat[:, : params.num_choice], k=k
        )
    return results


def preprocess_logits_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor, params: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess model logits and labels for metric computation.

    Args:
        logits: Model output logits
        labels: Ground truth labels
        params: Parameters containing choice tokens

    Returns:
        Processed logits and labels
    """
    logits = logits[:, -2][:, params.choice_tokens]
    labels = labels[:, -1]
    return logits, labels


def add_prompt(
    df: pd.DataFrame, mapping: pd.DataFrame, params: Any, is_train: bool = True
) -> pd.DataFrame:
    """
    Create dataset with misconception choices and labels.

    Args:
        df: Input DataFrame
        mapping: Misconception mapping DataFrame
        params: Configuration parameters
        is_train: Whether processing training data

    Returns:
        Processed DataFrame with prompts and labels
    """
    # Convert prediction IDs from string to list of integers
    df["pred_ids"] = df["pred_ids"].apply(lambda x: list(map(int, x.split())))

    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
        max_iterations = math.ceil((params.train_topk - params.num_choice) / params.num_slide) + 1

        for i in range(max_iterations):
            new_row = row.copy()

            # Calculate sliding window indices
            start_idx = params.num_slide * i
            end_idx = params.num_slide * i + params.num_choice

            # Adjust indices if they exceed train_topk
            if end_idx > params.train_topk:
                diff = end_idx - params.train_topk
                start_idx -= diff
                end_idx -= diff

            # Extract misconception IDs for current window
            mis_ids = row["pred_ids"][start_idx:end_idx]
            if len(mis_ids) != params.num_choice:
                raise ValueError(
                    f"Expected {params.num_choice} misconceptions, got {len(mis_ids)}"
                )

            # Get misconception names
            names = mapping.loc[mis_ids]["MisconceptionName"].tolist()

            # Determine label choice
            if row["MisconceptionId"] in mis_ids:
                new_row["label_choice"] = params.choice_words[
                    mis_ids.index(row["MisconceptionId"])
                ]
            else:
                # Skip if NA is not allowed or beyond negative topk in training
                if (not params.add_na or end_idx > params.train_negative_topk) and is_train:
                    continue
                new_row["label_choice"] = params.choice_words[-1]

            # Format misconception names with choice letters
            names = "\n".join(
                [f"{x}: {y}" for x, y in zip(params.choice_words[: params.num_choice], names)]
            )
            new_row["mis_names"] = names
            new_row["rank"] = f"{start_idx + 1}-{end_idx}"
            new_rows.append(new_row)

            # For validation, only process first window
            if not is_train:
                break

    # Create final DataFrame
    result_df = pd.DataFrame(new_rows)
    result_df["last_choice"] = params.choice_words[-1]

    # Apply appropriate prompt format
    result_df["prompt"] = result_df.apply(
        lambda x: NA_PROMPT_FORMAT.format(**x) if params.add_na else PROMPT_FORMAT.format(**x),
        axis=1,
    )
    return result_df


def main(
    fold: int = typer.Option(..., help="Fold number for cross-validation"),
    config: str = typer.Option("./config/exp_gpu.yaml", help="Path to configuration file"),
) -> None:
    """
    Main training function.

    Args:
        fold: Cross-validation fold number
        config: Path to configuration file
    """
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.train_listwise
    set_seed(params.seed)

    # Load data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    df = pd.read_csv(Path(cfg.save_dir) / params.input_name)

    # Split data into train and validation
    train_df = df.loc[df.fold != fold].copy()
    val_df = df.loc[(df.fold == fold) & (df.original)].copy()

    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=params.model_name,
        max_seq_length=params.max_length,
        dtype=torch.bfloat16,
        load_in_4bit=params.load_in_4bit,
        device_map="auto",
    )
    tokenizer.truncation_side = "left"

    # Setup choice words and tokens
    params.choice_words = get_choice_words(params.num_choice)
    params.choice_tokens = [tokenizer.encode(x)[0] for x in params.choice_words]

    # Process datasets
    train_df = add_prompt(train_df, mapping, params)
    val_df = add_prompt(val_df, mapping, params, is_train=False)
    train_dset = process_data(train_df, tokenizer, max_length=params.max_length)
    val_dset = process_data(val_df, tokenizer, max_length=params.max_length)

    # Plot token length distribution
    token_lengths = [len(x) for x in train_dset["input_ids"]]
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=100)
    plt.title("Distribution of Token Lengths")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.savefig("token_length_hist.png")
    plt.close()

    # Setup PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        **params["lora_config"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # Initialize wandb and setup paths
    run_name = f"fold_{fold}"
    wandb.init(project="eedi-reranker", name=f"{run_name}_{params.model_name.split('/')[-1]}")

    output_dir = str(Path(cfg.save_dir) / params.output_dir / run_name)
    best_model_path = str(Path(cfg.best_model_dir) / params.output_dir / run_name)

    # Calculate evaluation steps (4 times per epoch)
    steps_per_epoch = len(train_dset) // (
        params.train_args.per_device_train_batch_size
        * params.train_args.gradient_accumulation_steps
        * 4
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dset,
        eval_dataset=val_dset,
        dataset_text_field="prompt",
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest", max_length=params.max_length
        ),
        dataset_num_proc=4,
        packing=False,
        compute_metrics=partial(compute_metrics, params=params),
        preprocess_logits_for_metrics=partial(preprocess_logits_for_metrics, params=params),
        args=TrainingArguments(
            **params.train_args,
            output_dir=output_dir,
            eval_steps=steps_per_epoch,
            save_steps=steps_per_epoch,
        ),
    )

    # Apply response-only training
    trainer = train_on_responses_only(
        trainer,
        response_part="<|im_start|>assistant\n",
        instruction_part="<|im_start|>user\n",
    )

    # Train and save model
    trainer.train()
    trainer.save_model(best_model_path)


if __name__ == "__main__":
    typer.run(main)

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Set

import pandas as pd
import typer
from datasets import Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import mine_hard_negatives
from transformers import set_seed

import wandb

# Template for formatting the prompt
PROMPT_FORMAT: str = """Subject: {SubjectName}
Construct: {ConstructName}
Question: {QuestionText}
CorrectAnswer: {Correct}
IncorrectAnswer: {Answer}
IncorrectReason: {kd}"""


def create_val(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Create validation dataset by merging dataframe with mapping and adding labels.

    Args:
        df: Input DataFrame containing the base data
        mapping: DataFrame containing misconception mapping information

    Returns:
        DataFrame with processed validation data
    """
    df = df.merge(mapping, how="cross")
    df["label"] = 0
    df.loc[df["MisconceptionId_x"] == df["MisconceptionId_y"], "label"] = 1
    target_cols = ["prompt", "MisconceptionName_y", "label"]
    df = df[target_cols].rename(columns={"MisconceptionName_y": "MisconceptionName"})
    return df


def create_evaluator(df: pd.DataFrame, name: str = "train") -> InformationRetrievalEvaluator:
    """
    Create an evaluator for information retrieval tasks.

    Args:
        df: DataFrame containing prompts, misconception names, and labels
        name: Name identifier for the evaluator

    Returns:
        Configured InformationRetrievalEvaluator object
    """
    relevant_docs: DefaultDict[str, Set[str]] = defaultdict(set)
    queries: Dict[str, str] = {str(k): v for k, v in enumerate(df["prompt"].unique())}
    corpus: Dict[str, str] = {str(k): v for k, v in enumerate(df["MisconceptionName"].unique())}

    # Create reverse mappings for efficient lookup
    qid_dict: Dict[str, str] = {v: k for k, v in queries.items()}
    cid_dict: Dict[str, str] = {v: k for k, v in corpus.items()}

    # Build relevant documents mapping
    for prompt, g in df.groupby("prompt"):
        for mis_name, label in g[["MisconceptionName", "label"]].values:
            if label == 1:
                qid = qid_dict[str(prompt)]
                cid = cid_dict[mis_name]
                relevant_docs[qid].add(cid)

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        map_at_k=[25],
        mrr_at_k=[25],
        precision_recall_at_k=[50, 100, 150, 200],
        ndcg_at_k=[25],
        accuracy_at_k=[25],
    )


def main(
    fold: int = typer.Option(..., help="Fold number for cross-validation"),
    config: str = typer.Option("./config/exp_gpu.yaml", help="Path to configuration file"),
) -> None:
    """
    Main training function for the bi-encoder model.

    Args:
        fold: Cross-validation fold number
        config: Path to configuration file
    """
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.train_biencoder
    set_seed(params.seed)

    # Load and prepare data
    df = pd.read_csv(Path(cfg.save_dir) / params.input_name)
    mapping = pd.read_csv("input/misconception_mapping.csv")
    df["prompt"] = df.apply(lambda x: PROMPT_FORMAT.format(**x), axis=1)

    # Split data into train and validation sets
    train_df = df.loc[df.fold != fold].copy()
    val_df = df.loc[(df.fold == fold) & (df.original)].copy()
    val_df = create_val(val_df, mapping)

    # Create dataset for training
    train_dset = Dataset.from_dict(
        {
            "anchor": train_df["prompt"].tolist(),
            "positive": train_df["MisconceptionName"].tolist(),
        }
    )

    # Setup paths and wandb
    name = f"fold_{fold}"
    output_dir = str(Path(cfg.save_dir) / params.output_dir / name)
    best_model_path = str(Path(cfg.best_model_dir) / params.output_dir / name)
    wandb.init(project="eedi-biencoder", name=f"{name}_{params.model_name.split('/')[-1]}")

    # Initialize model
    model = SentenceTransformer(
        params.model_name,
        trust_remote_code=True,
        model_kwargs={"load_in_4bit": params.load_in_4bit},
    )

    # Perform hard negative mining
    train_dset = mine_hard_negatives(
        train_dset,
        model,
        **params["hard_negative_params"],
    )

    # Add LoRA adapter if specified
    if params.is_lora:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            target_modules="all-linear",
            lora_dropout=0.01,
            **params["lora_config"],
        )
        model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
        model[0].auto_model.print_trainable_parameters()

    # Setup loss function and evaluator
    loss = losses.CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=params.mini_batch_size, show_progress_bar=True
    )
    val_evaluator = create_evaluator(val_df, name="val")

    # Configure training arguments
    args = SentenceTransformerTrainingArguments(
        **params.train_args,
        output_dir=output_dir,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train_dset,
        loss=loss,
        evaluator=val_evaluator,
    )
    trainer.train()
    trainer.save_model(best_model_path)


if __name__ == "__main__":
    typer.run(main)

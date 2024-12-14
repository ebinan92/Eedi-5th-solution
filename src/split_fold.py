from pathlib import Path
from typing import Set

import pandas as pd
import typer
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold
from transformers import set_seed

from utils import load_data


def calculate_misconception_overlap(train_misid: Set[int], val_misid: Set[int]) -> float:
    """Calculate the overlap ratio between training and validation misconceptions.

    Args:
        train_misid: Set of misconception IDs in training set
        val_misid: Set of misconception IDs in validation set

    Returns:
        float: Overlap ratio (1 - intersection/validation size)
    """
    return 1.0 - len(train_misid & val_misid) / len(val_misid)


def split_dataset(df: pd.DataFrame, n_splits: int, group_col: str) -> pd.DataFrame:
    """Split dataset into folds using GroupKFold.

    Args:
        df: Input DataFrame
        n_splits: Number of splits
        group_col: Column name to use for grouping

    Returns:
        pd.DataFrame: DataFrame with added fold column
    """
    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df[group_col])):
        if group_col == "QuestionId":
            # Calculate overlap for original data
            train_misid = set(df.loc[train_idx]["MisconceptionId"])
            val_misid = set(df.loc[val_idx]["MisconceptionId"])
            overlap = calculate_misconception_overlap(train_misid, val_misid)
            print(f"Fold {fold_idx} misconception overlap: {overlap:.3f}")

        df.loc[val_idx, "fold"] = fold_idx

    return df


def main(config: str = "./config/exp_gpu.yaml") -> None:
    """Split data into folds for cross-validation.

    Args:
        config: Path to configuration YAML file
    """
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.split_fold
    set_seed(params.seed)

    # Load misconception mapping
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")

    # Process original training data
    train_df = pd.read_csv(Path(cfg.input_dir) / "train.csv")
    df = load_data(train_df).reset_index(drop=True)
    df["original"] = True
    df = split_dataset(df, params.n_split, "QuestionId")

    # Process synthetic data
    synthetic_df = pd.read_csv(Path(cfg.save_dir) / params.add_data_name).reset_index(drop=True)
    synthetic_df["original"] = False
    synthetic_df = split_dataset(synthetic_df, params.n_split, "SubjectName")

    # Combine datasets
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    combined_df["fold"] = combined_df["fold"].astype(int)

    # Validate number of folds
    if combined_df["fold"].nunique() != params.n_split:
        raise ValueError(f"Expected {params.n_split} folds, got {combined_df['fold'].nunique()}")

    # Merge with misconception mapping and save
    final_df = combined_df.merge(mapping, on="MisconceptionId")
    output_path = Path(cfg.save_dir) / params.save_name
    final_df.to_csv(output_path, index=False)
    print(f"Saved split dataset to {output_path}")


if __name__ == "__main__":
    typer.run(main)

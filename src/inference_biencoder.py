import gc
from pathlib import Path
from typing import List

import pandas as pd
import torch
import typer
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from sentence_transformers import SentenceTransformer

from utils import print_scores

# Prompt template for formatting the input data
PROMPT_FORMAT: str = """Subject: {SubjectName}\nConstruct: {ConstructName}\nQuestion: {QuestionText}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}"""


def load_model(
    params: DictConfig,
    cfg: DictConfig,
    fold: int,
) -> SentenceTransformer:
    """
    Load the SentenceTransformer model with or without LoRA weights.

    Args:
        params: Parameters for model inference
        cfg: Configuration object
        fold: Current fold number

    Returns:
        Loaded SentenceTransformer model
    """
    if params.is_lora:
        model = SentenceTransformer(
            params.model_name,
            trust_remote_code=True,
        )
        lora_weight_path = str(Path(cfg.best_model_dir) / params.model_output_dir / f"fold_{fold}")
        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, lora_weight_path)
        model[0].auto_model.eval()
    else:
        model_name = str(Path(cfg.best_model_dir) / params.model_output_dir / f"fold_{fold}")
        model = SentenceTransformer(model_name, trust_remote_code=True)

    return model


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> torch.Tensor:
    """
    Encode texts using the SentenceTransformer model.

    Args:
        model: SentenceTransformer model
        texts: List of texts to encode
        batch_size: Batch size for encoding

    Returns:
        Tensor of encoded texts
    """
    return model.encode(
        texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )


def process_fold(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    model: SentenceTransformer,
    params: DictConfig,
    fold: int,
) -> pd.DataFrame:
    """
    Process a single fold of data.

    Args:
        df: Input DataFrame
        mapping: Misconception mapping DataFrame
        model: SentenceTransformer model
        params: Parameters for model inference
        fold: Current fold number

    Returns:
        Processed DataFrame for the current fold
    """
    val_df = df.loc[df.fold == fold].copy()

    # Encode texts
    mapping_encoddings = encode_texts(
        model,
        mapping["MisconceptionName"].tolist(),
        params.batch_size,
    )
    prompt_encoddings = encode_texts(
        model,
        val_df["prompt"].tolist(),
        params.batch_size,
    )

    # Calculate similarities and get predictions
    similarity = model.similarity(prompt_encoddings, mapping_encoddings).cpu().numpy()
    indices = similarity.argsort()[:, ::-1]
    val_df["pred_ids"] = [" ".join(map(str, idxs)) for idxs in indices]

    # Print evaluation scores
    print_scores(val_df[val_df.original], k_list=[25, 50, 100, 150])

    return val_df


def main(config: str = "./config/exp_gpu.yaml") -> None:
    """
    Main function to run the inference process.

    Args:
        config: Path to the configuration file
    """
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.inference_biencoder

    # Load input data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    df = pd.read_csv(Path(cfg.save_dir) / params.input_name)

    # Format prompts
    df["prompt"] = df.apply(
        lambda x: PROMPT_FORMAT.format(**x),
        axis=1,
    )

    processed_dfs: List[pd.DataFrame] = []
    n_split = cfg.split_fold.n_split

    # Process each fold
    for fold in range(n_split):
        model = load_model(params, cfg, fold)
        val_df = process_fold(df, mapping, model, params, fold)
        processed_dfs.append(val_df)

        # Clean up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    pd.concat(processed_dfs).to_csv(Path(cfg.save_dir) / params.save_name, index=False)


if __name__ == "__main__":
    typer.run(main)

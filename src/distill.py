from pathlib import Path
from typing import List

import pandas as pd
import typer
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
from vllm import LLM, SamplingParams

# Set random seed for reproducibility
set_seed(42)

# Template for generating prompts
PROMPT_FORMAT: str = """<|im_start|>system
You will be given a math problem and its correct and incorrect answer.
First explain why the correct answer is correct, and finally explain reasons and misconceptions for incorrect answer.
Please briefly explain in 200 words or less.<|im_end|>
<|im_start|>user
Problem: {QuestionText}\nCorrect Answer: {Correct}\nIncorrect Answer: {Answer}.<|im_end|>
<|im_start|>assistant
"""


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        DictConfig: Loaded configuration
    """
    try:
        config = OmegaConf.load(config_path)
        if not isinstance(config, DictConfig):
            raise ValueError(f"Config loaded from {config_path} is not a DictConfig")
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {str(e)}")


def create_model(params: DictConfig) -> LLM:
    """
    Initialize the LLM model with specified parameters.

    Args:
        params (DictConfig): Model configuration parameters

    Returns:
        LLM: Initialized model
    """
    return LLM(
        model=params.model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.99,
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,
    )


def generate_prompts(df: pd.DataFrame) -> List[str]:
    """
    Generate prompts from DataFrame using the template.

    Args:
        df (pd.DataFrame): Input DataFrame containing question data

    Returns:
        List[str]: List of formatted prompts
    """
    return df.apply(lambda x: PROMPT_FORMAT.format(**x), axis=1).tolist()


def generate_responses(model: LLM, prompts: List[str]) -> List[str]:
    """
    Generate responses using the model for given prompts.

    Args:
        model (LLM): Initialized LLM model
        prompts (List[str]): List of input prompts

    Returns:
        List[str]: Generated responses
    """
    sampling_params = SamplingParams(
        max_tokens=2048,
        stop=["<|im_end|>"],
        temperature=0.0,
    )
    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def main(
    config: str = "./config/exp_gpu.yaml",
) -> None:
    """
    Main function to run the distillation process.

    Args:
        config (str): Path to configuration file
    """
    try:
        # Load configuration
        cfg = load_config(config)
        params = cfg.distill

        # Read input data
        input_path = Path(cfg.save_dir) / params.input_name
        df = pd.read_csv(input_path)

        # Generate prompts
        df["prompt"] = generate_prompts(df)

        # Initialize model and generate responses
        model = create_model(params)
        df["kd"] = generate_responses(model, df.prompt.tolist())

        # Save results
        output_path = Path(cfg.save_dir) / params.save_name
        df.to_csv(output_path, index=False)

    except Exception as e:
        raise RuntimeError(f"Distillation process failed: {str(e)}")


if __name__ == "__main__":
    typer.run(main)

import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Tuple

import pandas as pd
import torch
import typer
from dotenv import load_dotenv
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio

from utils import load_data

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


async def get_completion(messages: List[Dict[str, str]], model_name: str) -> Dict[str, str]:
    """
    Get completion from OpenAI API with retry logic.

    Args:
        messages: List of message dictionaries
        model_name: Name of the model to use

    Returns:
        Dictionary containing the completion response
    """
    n_try = 0
    tmp = {"Incorrect": "", "Correct": ""}

    def escape_backslashes(s: str) -> str:
        """Escape backslashes in string."""
        return s.replace("\\", "\\\\")

    while True:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            output = response.choices[0].message.content.strip()
            output = escape_backslashes(output)
            output = output.replace("\n", "\\n")
            output = json.loads(output)

            # Retry if incorrect and correct answers are the same
            if output["Incorrect"] == output["Correct"]:
                if n_try < 3:
                    tmp = output
                    n_try += 1
                    continue
            return output

        except Exception as e:
            n_try += 1
            if n_try >= 3:
                print("Failed to get completion")
                print(e)
                print(output)
                return tmp


def create_prompt(row: pd.Series, example_rows: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Create prompt for completion API with examples.

    Args:
        row: Row containing misconception information
        example_rows: DataFrame containing example rows

    Returns:
        List containing the formatted prompt
    """

    def clean_text(text: str) -> str:
        """Remove leading/trailing whitespace and dedent text."""
        return dedent(text).strip()

    system_prompt = {
        "role": "user",
        "content": clean_text("""
        Your task is to generate a math problem and correct and a example of incorrect answer for that problem.
        You will be given only the wrong reason of incorrect answer, so please generate from there. Specifically, please generate the following content:

        # Output format
        QuestionText: Math question
        ConstructName: Most granular level of knowledge related to question
        SubjectName: More general context than the ConstructName
        Incorrect: incorrect answer for given wrong reason
        Correct: correct answer

        Please provide the output in the following valid JSON format:
        {{"QuestionText": str, "ConstructName": str, "SubjectName": str, "Incorrect": str, "Correct": str}}

        # Example
        {example}
        
        # Target
        Wrong Reason: {MisconceptionName}
        """),
    }

    example_format = clean_text("""
        Wrong Reason: {MisconceptionName}
        {{"QuestionText": "{QuestionText}", "ConstructName": "{ConstructName}", "SubjectName": "{SubjectName}", "Incorrect": "{Answer}", "Correct": "{Correct}"}}
    """)

    # Convert double quotes to single quotes
    cols_to_convert = ["QuestionText", "ConstructName", "SubjectName", "Answer", "Correct"]
    example_rows[cols_to_convert] = example_rows[cols_to_convert].map(
        lambda x: x.replace('"', "'")
    )

    example = "\n\n".join(
        [example_format.format(**example_row) for _, example_row in example_rows.iterrows()]
    )
    system_prompt["content"] = system_prompt["content"].format(
        example=example, MisconceptionName=row["MisconceptionName"]
    )
    return [system_prompt]


async def run_generation(
    messages: List[List[Dict[str, str]]], model_name: str, sem_number: int
) -> List[Dict[str, str]]:
    async def sem_execution(
        prompt: List[Dict[str, str]], model_name: str, sem_number: int
    ) -> Dict[str, str]:
        sem = asyncio.Semaphore(sem_number)
        """Execute API call with semaphore for concurrent request limiting."""
        async with sem:
            return await get_completion(prompt, model_name)

    """Run multiple completion requests concurrently."""
    outputs = [sem_execution(prompt, model_name, sem_number) for prompt in messages]
    return await tqdm_asyncio.gather(*outputs)


def create_input_data(
    target_mapping: pd.DataFrame,
    fewshot_df: pd.DataFrame,
    emb_model: SentenceTransformer,
    num_shot: int = 3,
) -> Tuple[List[List[Dict[str, str]]], List[int]]:
    """
    Create input data for completion API with similar examples.

    Args:
        target_mapping: DataFrame containing target misconceptions
        fewshot_df: DataFrame containing example data
        emb_model: Sentence transformer model for embedding
        num_shot: Number of examples to include

    Returns:
        Tuple of messages and misconception IDs
    """

    def embed_text(text_list: List[str], emb_model: SentenceTransformer) -> torch.Tensor:
        """Embed text using sentence transformer model."""
        mis_embs = emb_model.encode(
            text_list,
            convert_to_tensor=True,
            batch_size=32,
        )
        return mis_embs.cpu()

    messages = []
    misconception_ids = []

    mis_names = fewshot_df["MisconceptionName"].tolist()
    mis_embs = embed_text(mis_names, emb_model)

    for _, row in target_mapping.iterrows():
        mis_target_emb = embed_text([row["MisconceptionName"]], emb_model)
        similarity = emb_model.similarity(mis_target_emb, mis_embs).cpu().numpy()
        indices = similarity.argsort()[:, ::-1][:, :num_shot][0]
        g = fewshot_df.iloc[indices].copy()
        message = create_prompt(row, g)
        messages.append(message)
        misconception_ids.append(row["MisconceptionId"])

    return messages, misconception_ids


def main(config: str = "./config/exp_gpu.yaml") -> None:
    """
    Main function to generate questions using completion API.

    Args:
        config: Path to configuration file
    """
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.generate_question

    # Initialize embedding model
    emb_model = SentenceTransformer(
        params.embedding_model,
        trust_remote_code=True,
    )

    # Create save directory if it doesn't exist
    if not os.path.exists(Path(cfg.save_dir)):
        os.makedirs(Path(cfg.save_dir))

    # Load and prepare data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    train = pd.read_csv(Path(cfg.input_dir) / "train.csv")
    df = load_data(train)
    df = df.merge(mapping, on="MisconceptionId")
    fewshot_df = df.groupby("MisconceptionId").sample(1)

    messages = []
    misconception_ids = []

    # Process non-misconception samples
    target_mapping = mapping.loc[~mapping.MisconceptionId.isin(df.MisconceptionId.unique())].copy()
    ms, m_ids = create_input_data(
        target_mapping, fewshot_df, emb_model=emb_model, num_shot=params.num_shot
    )
    messages.extend(ms)
    misconception_ids.extend(m_ids)
    print("non misconception sample", len(ms))

    # Generate completions
    outputs = asyncio.run(run_generation(messages, params.model_name, params.sem_number))

    # Filter and process outputs
    target_key = [
        "QuestionText",
        "ConstructName",
        "SubjectName",
        "Incorrect",
        "Correct",
    ]

    valid_outputs = [
        (out, mid)
        for out, mid in zip(outputs, misconception_ids)
        if set(out.keys()) == set(target_key)
    ]

    if valid_outputs:
        new_outputs, new_misconception_ids = zip(*valid_outputs)
    else:
        new_outputs, new_misconception_ids = [], []

    # Create and save output DataFrame
    pred_df = pd.DataFrame(new_outputs)
    pred_df["MisconceptionId"] = new_misconception_ids
    pred_df.rename(columns={"Incorrect": "Answer"}, inplace=True)
    pred_df.to_csv(Path(cfg.save_dir) / params.save_name, index=False)


if __name__ == "__main__":
    typer.run(main)

from pprint import pprint
from typing import Dict, List, Union

import numpy as np
import pandas as pd


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame to create a structured dataset for misconception analysis.

    Args:
        df: Input DataFrame containing answer and misconception data

    Returns:
        Processed DataFrame with restructured data
    """
    datas: List[Dict[str, Union[str, int]]] = []
    for col in ["A", "B", "C", "D"]:
        answer_col = f"Answer{col}Text"
        misconception_col = f"Misconception{col}Id"

        for _, row in df.iterrows():
            if pd.isna(row[misconception_col]):
                continue

            correct_col = row["CorrectAnswer"]
            correct_answer_col = f"Answer{correct_col}Text"

            data = {
                "Answer": row[answer_col],
                "Correct": row[correct_answer_col],
                "MisconceptionId": int(row[misconception_col]),
            }

            if correct_col == col:
                continue

            row_dict = row.to_dict()
            # Remove unnecessary columns
            for c in ["A", "B", "C", "D"]:
                row_dict.pop(f"Misconception{c}Id")
                row_dict.pop(f"Answer{c}Text")

            data.update(row_dict)
            datas.append(data)

    return pd.DataFrame(datas)


def average_precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 25) -> float:
    """
    Calculate Average Precision at K for individual samples.

    Args:
        scores: Array of prediction scores
        labels: Array of true labels (0 or 1)
        k: Number of top items to consider

    Returns:
        Average Precision at K score
    """
    topk_indices = np.argsort(-scores)[:k]
    topk_labels = labels[topk_indices]

    relevant = topk_labels == 1
    if not np.any(relevant):
        return 0.0

    precisions: List[float] = []
    num_relevant = 0
    for i in range(k):
        if relevant[i]:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precisions.append(precision_at_i)

    return float(np.mean(precisions)) if precisions else 0.0


def mean_average_precision_at_k(
    prediction_scores: np.ndarray, labels: np.ndarray, k: int = 25
) -> float:
    """
    Calculate Mean Average Precision at K for all samples in batch.

    Args:
        prediction_scores: Array of prediction scores for batch
        labels: Array of true labels for batch
        k: Number of top items to consider

    Returns:
        Mean Average Precision at K score
    """
    average_precisions: List[float] = []
    batch_size = prediction_scores.shape[0]

    for i in range(batch_size):
        ap = average_precision_at_k(prediction_scores[i], labels[i], k)
        average_precisions.append(ap)

    return float(np.mean(average_precisions))


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 25) -> float:
    """
    Calculate Recall at K for individual samples.

    Args:
        scores: Array of prediction scores
        labels: Array of true labels (0 or 1)
        k: Number of top items to consider

    Returns:
        Recall at K score
    """
    topk_indices = np.argsort(-scores)[:k]
    topk_labels = labels[topk_indices]

    num_relevant_at_k = np.sum(topk_labels == 1)
    total_relevant = np.sum(labels == 1)

    if total_relevant == 0:
        return 0.0

    return num_relevant_at_k / total_relevant


def mean_recall_at_k(prediction_scores: np.ndarray, labels: np.ndarray, k: int = 25) -> float:
    """
    Calculate Mean Recall at K for all samples in batch.

    Args:
        prediction_scores: Array of prediction scores for batch
        labels: Array of true labels for batch
        k: Number of top items to consider

    Returns:
        Mean Recall at K score
    """
    recalls: List[float] = []
    batch_size = prediction_scores.shape[0]

    for i in range(batch_size):
        recall = recall_at_k(prediction_scores[i], labels[i], k)
        recalls.append(recall)

    return float(np.mean(recalls))


def calculate_map_at_k(predictions: List[int], ground_truth: List[int], k: int = 25) -> float:
    """
    Calculate MAP@K (Mean Average Precision at K).

    Args:
        predictions: List of recommended item IDs
        ground_truth: List of correct item IDs
        k: Number of top items to consider

    Returns:
        MAP@K score
    """
    if not ground_truth:
        return 0.0

    predictions = predictions[:k]
    score = 0.0
    num_hits = 0.0

    for i, pred in enumerate(predictions):
        if pred in ground_truth:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(ground_truth), k)


def calculate_recall_at_k(predictions: List[int], ground_truth: List[int], k: int = 25) -> float:
    """
    Calculate Recall@K.

    Args:
        predictions: List of recommended item IDs
        ground_truth: List of correct item IDs
        k: Number of top items to consider

    Returns:
        Recall@K score
    """
    if not ground_truth:
        return 0.0

    predictions = predictions[:k]
    num_hits = len(set(predictions) & set(ground_truth))

    return num_hits / len(ground_truth)


def print_scores(df: pd.DataFrame, k_list: List[int] = [25, 50]) -> None:
    """
    Print evaluation scores for different K values.

    Args:
        df: DataFrame containing predictions and ground truth
        k_list: List of K values to evaluate
    """
    tmp_df = df.copy()
    for k in k_list:
        tmp_df[f"recall@{k}"] = tmp_df.apply(
            lambda x: calculate_recall_at_k(
                list(map(int, x["pred_ids"].split())), [x["MisconceptionId"]], k=k
            ),
            axis=1,
        )
        tmp_df[f"map@{k}"] = tmp_df.apply(
            lambda x: calculate_map_at_k(
                list(map(int, x["pred_ids"].split())), [x["MisconceptionId"]], k=k
            ),
            axis=1,
        )

    result_columns = [f"recall@{k}" for k in k_list] + [f"map@{k}" for k in k_list]
    pprint(tmp_df[result_columns].mean().to_dict())

import json
import os
from typing import Dict, List

from src.field_retrieval import (
    DescriptionRetrieval,
    ExamplesRetrieval,
    ParametersRetrieval,
    ResponseRetrieval,
)
from src.utils import get_augment_tools_file, trec_eval, write_data


def load_train_test_and_tools(
    dataset: str,
    fold: int,
    hard_negatives_from: str = "bm25_origin",
):
    """Load train and test data for a given dataset and fold.

    Args:
        dataset (str): The name of the dataset.
        fold (int): The fold number.

    Returns:
        tuple: A tuple containing the training data and testing data.
    """
    assert 0 <= fold < 5, "fold number must be between 0 and 4."
    base_path = os.path.join("dataset", "train_and_test", dataset, f"fold_{fold}")
    with open(os.path.join(base_path, "train_queries.json"), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(base_path, "test_queries.json"), "r") as f:
        test_data = json.load(f)
    # 检查 train 和 test 是否完全没有重叠
    train_ids = set(item["id"] for item in train_data)
    test_ids = set(item["id"] for item in test_data)
    assert train_ids.isdisjoint(test_ids), (
        "Train and test data have overlapping IDs: {} ".format(
            train_ids.intersection(test_ids)
        )
    )
    train_queries = set(item["query"] for item in train_data)
    test_queries = set(item["query"] for item in test_data)
    assert train_queries.isdisjoint(test_queries), (
        "Train and test data have overlapping queries: {} ".format(
            train_queries.intersection(test_queries)
        )
    )

    tools = json.load(open(get_augment_tools_file(dataset), "r"))

    # load hard negatives
    hard_neg_file = os.path.join(
        "dataset",
        "hard_negatives",
        hard_negatives_from,
        f"{dataset}.json",
    )
    hard_neg = json.load(open(hard_neg_file, "r"))

    return {
        "train": train_data,
        "test": test_data,
        "tools": tools,
        "hard_negatives": hard_neg,
    }


def calc_field_scores(
    dataset: str,
    queries,
    tools,
    model_name: str,
    batch_size: int,
    redo: bool = False,
    retrieve_fields: List[str] = ["description", "parameters", "response", "examples"],
    tmp_save_root_dir: str = None,
):
    """
    Calculate field scores for the given queries and tools using a specified model.
    """
    if tmp_save_root_dir is None:
        tmp_save_root_dir = "tmp_field_retrieval_results"

    field_to_retriever_class = {
        "description": DescriptionRetrieval,
        "parameters": ParametersRetrieval,
        "response": ResponseRetrieval,
        "examples": ExamplesRetrieval,
    }
    results = {}
    for field in retrieve_fields:
        retriever = field_to_retriever_class[field](
            model_name=model_name,
            batch_size=batch_size,
            redo=redo,
        )
        if redo:
            results[field] = retriever.retrieve(queries, tools)["retrieval_results"]
        else:
            tmp_save_dir = os.path.join(
                tmp_save_root_dir,
                dataset,
                field,
                model_name.split("/")[-1] + "__bs_" + str(batch_size),
            )
            os.makedirs(tmp_save_dir, exist_ok=True)

            def save_or_compare(file_name, data):
                file_path = os.path.join(tmp_save_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        existing_data = json.load(f)
                    existing_data = sorted(existing_data, key=lambda x: x["id"])
                    data = sorted(data, key=lambda x: x["id"])
                    assert existing_data == data, f"Data mismatch in {file_path}"
                else:
                    write_data(data, file_path)

            save_or_compare("queries.json", queries)
            save_or_compare("tools.json", tools)
            results[field] = retriever.retrieve_and_save(
                queries, tools, save_dir=tmp_save_dir
            )["details"]
    return results


def eval_test_results(
    test_queries: List[Dict],
    pred_details: Dict[str, Dict[str, float]],
):
    """
    pred_details[query_id][tool_id] = score
    """
    qrels = {}
    for item in test_queries:
        qrels[item["id"]] = {str(x["id"]): int(x["relevance"]) for x in item["labels"]}
    collection = trec_eval(qrels=qrels, results=pred_details)
    return collection

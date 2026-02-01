import argparse
import json
import os
import random

import numpy as np
from src.utils import (
    eval_retrieval,
    get_rewrite_queries_file,
    load_local_data,
    write_data,
)

# train_ratio = 0.8
seed = 42
random.seed(seed)
np.random.seed(seed)

dataset_list = ["apibank", "gorilla-huggingface", "toolink", "apigen", "toolbench"]
print(dataset_list)


def split_train_test(dataset_name):
    queries, _ = load_local_data(dataset_name)
    queries = json.load(open(get_rewrite_queries_file(dataset_name)))
    print("Total queries:", len(queries))
    random.shuffle(queries)
    for i in range(5):
        test_queries = queries[i * len(queries) // 5 : (i + 1) * len(queries) // 5]
        train_queries = (
            queries[: i * len(queries) // 5] + queries[(i + 1) * len(queries) // 5 :]
        )
        assert (
            len(
                set(q["id"] for q in train_queries).intersection(
                    set(q["id"] for q in test_queries)
                )
            )
            == 0
        )
        output_dir = f"./dataset/train_and_test/{dataset_name}/fold_{i}"
        os.makedirs(output_dir, exist_ok=True)

        write_data(train_queries, os.path.join(output_dir, "train_queries.json"))
        write_data(test_queries, os.path.join(output_dir, "test_queries.json"))

        print(
            f"Fold {i}: Train queries: {len(train_queries)}, Test queries: {len(test_queries)}"
        )


def get_hard_negatives_by_bm25(dataset_name):
    queries, tools = load_local_data(dataset_name)
    ret_details = eval_retrieval(
        queries=queries,
        tools=tools,
        model_name="bm25",
        top_k=150,
    )["details"]
    # 提取前 100 个不在 labels 中的 tool 作为 hard negatives
    hard_negatives = {}
    for query in queries:
        query_id = query["id"]
        label_ids = [t["id"] for t in query.get("labels", [])]
        neg_tools = []
        for tid, score in ret_details[query_id].items():
            if tid not in label_ids:
                neg_tools.append((tid, score))
            if len(neg_tools) >= 100:
                break
        hard_negatives[query_id] = neg_tools
    output_dir = "./dataset/hard_negatives/bm25_origin/"
    os.makedirs(output_dir, exist_ok=True)
    write_data(
        hard_negatives,
        os.path.join(output_dir, f"{dataset_name}.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    args = parser.parse_args()
    split_train_test(args.dataset)
    get_hard_negatives_by_bm25(args.dataset)

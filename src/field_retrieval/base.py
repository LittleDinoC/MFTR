import json
import os
from typing import Dict, List

import bm25s
import faiss
import numpy as np
from src.utils import (
    USE_RETRIEVER_MODELS_AND_BATCH_SIZE,
    write_data,
)
from toolret.eval import RetModel


def get_default_batch_size(model_name):
    for model, batch_size in USE_RETRIEVER_MODELS_AND_BATCH_SIZE:
        if model.startswith(model_name) or model.endswith(model_name):
            return batch_size
    else:
        return 2


def run_retrieval(
    queries: List[Dict],
    tools: List[Dict],
    model_name: str,
    batch_size: int = None,
    top_k: int = None,
):
    """
    queries: 至少包含 "id" 和 "query"，"query" 将被用于检索
    tools: 至少包含 "id" 和 "documentation"，"documentation" 将被 embedded 用于检索，必须是 string

    返回格式:
    {
        "query_id1": {
            "tool_id1": score1,
            "tool_id2": score2,
            ...
        },
    }
    """
    if batch_size is None:
        batch_size = get_default_batch_size(model_name)
    if top_k is None:
        top_k = len(tools)
    top_k = min(top_k, len(tools))

    if model_name == "bm25":
        doc_content = [tool["documentation"] for tool in tools]
        doc_ids = [tool["id"] for tool in tools]
        corpus_tokens = bm25s.tokenize(doc_content, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        results = {}
        for item in queries:
            query = item["query"]
            query_tokens = bm25s.tokenize(query)
            if len(query_tokens.vocab) == 0:
                query_tokens = bm25s.tokenize("NONE", stopwords=[])
            hits, scores = retriever.retrieve(
                query_tokens, corpus=doc_ids, k=min(top_k, len(doc_ids))
            )
            results[item["id"]] = {}
            for hit, score in zip(hits[0], scores[0]):
                results[item["id"]][str(hit)] = float(score)

    else:
        model = RetModel(
            model_name,
        )
        tool_embeddings = model.encode_tools(tools, batch_size)
        tool_embeddings = np.asarray(tool_embeddings, dtype=np.float32)
        dim = tool_embeddings.shape[1]
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(tool_embeddings)
        print("Encoding query embeddings...")
        query_embeddings = model.encode_queries(queries, batch_size)
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        distance, rank = index.search(query_embeddings, top_k)
        results = {}
        for item, rk, ds in zip(queries, rank, distance):
            results[item["id"]] = {}
            for r, d in zip(rk, ds):
                results[item["id"]][str(tools[int(r)]["id"])] = float(d)

    return results


class FieldRetrievalBase:
    def __init__(
        self,
        model_name: str,
        batch_size: int = None,
        top_k: int = None,
        redo: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.redo = redo

    def preprocess_data(
        self,
        raw_queries: List[Dict],  # rewritten queries
        raw_tools: List[Dict],  # augmented tools
    ):
        return raw_queries, raw_tools

    def retrieve_processed_data(
        self,
        processed_queries: List[Dict],
        processed_tools: List[Dict],
    ):
        return run_retrieval(
            queries=processed_queries,
            tools=processed_tools,
            model_name=self.model_name,
            batch_size=self.batch_size,
            top_k=self.top_k,
        )

    def retrieve(
        self,
        queries: List[Dict],
        tools: List[Dict],
    ):
        processed_queries, processed_tools = self.preprocess_data(queries, tools)
        return {
            "processed_queries": processed_queries,
            "processed_tools": processed_tools,
            "retrieval_results": self.retrieve_processed_data(
                processed_queries, processed_tools
            ),
        }

    def retrieve_and_save(
        self,
        queries: List[Dict],
        tools: List[Dict],
        save_dir: str = None,
    ):
        """
        {
            "id": "query_1",
            "query": "What is ...?",
        }
        """
        if (
            self.redo is False
            and save_dir is not None
            and os.path.exists(os.path.join(save_dir, "details.json"))
        ):
            print(f"Loading existing results from {save_dir}...")
            details = json.load(open(os.path.join(save_dir, "details.json"), "r"))
            return {"details": details}

        _retrieval_results = self.retrieve(queries, tools)
        processed_queries = _retrieval_results["processed_queries"]
        processed_tools = _retrieval_results["processed_tools"]
        retrieval_results = _retrieval_results["retrieval_results"]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            configs = {
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "top_k": self.top_k,
                "raw_queries": queries,
                "raw_tools": tools,
                "processed_queries": processed_queries,
                "processed_tools": processed_tools,
            }
            write_data(configs, os.path.join(save_dir, "configs.json"))
            write_data(retrieval_results, os.path.join(save_dir, "details.json"))
        return {"details": retrieval_results}

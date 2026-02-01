import json
import os
import pickle
from copy import deepcopy
from typing import Dict, List, Tuple

import bm25s
import faiss
import numpy as np
import yaml
from src.llms import LlmFactory
from termcolor import colored
from toolret.eval import RetModel, trec_eval

USE_RETRIEVER_MODELS_AND_BATCH_SIZE = [
    ("bm25", 0),
    ("sentence-transformers/all-MiniLM-L6-v2", 16),
    ("facebook/contriever-msmarco", 16),
    ("intfloat/multilingual-e5-base", 16),
    ("intfloat/multilingual-e5-large", 16),
    ("Alibaba-NLP/gte-large-en-v1.5", 4),
    ("BAAI/bge-large-en-v1.5", 16),
    ("Tool-COLT/contriever-base-msmarco-v1-ToolBenchG3", 16),
    ("ToolBench/ToolBench_IR_bert_based_uncased", 16),
]


def load_local_data(dataset_name) -> Tuple[List[Dict], List[Dict]]:
    """
    Load local queries and tools from JSON files.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Tuple[List[Dict], List[Dict]]: A tuple containing a list of queries and a list of tools.
    """
    dataset_dir = f"./dataset/from_toolret/{dataset_name}"
    print(f"Loading dataset {dataset_name} from {dataset_dir}...")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    queries = json.load(open(os.path.join(dataset_dir, "queries.json"), "r"))
    tools = json.load(open(os.path.join(dataset_dir, "tools.json"), "r"))
    return queries, tools


def get_augment_tools_file(
    dataset_name: str,
    augment_llm: str = "gpt-4o-mini",
):
    augment_llm = get_llm_name(augment_llm)
    return f"dataset/augmented_tools/{dataset_name}/{augment_llm}_tools.json"


def get_rewrite_queries_file(
    dataset_name: str,
    augment_and_rewrite_llm: str = "gpt-4o-mini",
    base_retriever: str = "bm25",
    top_k: int = 20,
    prompt_file: str = "query_rewrite",
):
    augment_and_rewrite_llm = get_llm_name(augment_and_rewrite_llm)
    file_dir = f"dataset/rewritten_queries/{dataset_name}/augment_llm_{augment_and_rewrite_llm}__retriever_{base_retriever}__topk_{top_k}__prompt_{prompt_file}/"
    return file_dir + "queries.json"


def eval_retrieval(
    queries: List[Dict],
    tools: List[Dict],
    model_name: str,
    batch_size: int = 4,
    top_k: int = 100,
    save_dir: str = None,
) -> Dict[str, Dict]:
    tools = deepcopy(tools)
    print(f"Total tools: {len(tools)}")
    for tool in tools:
        assert tool["documentation"] is not None, (
            f"Tool {tool['id']} has no documentation"
        )
        if type(tool["documentation"]) is not str:
            tool["documentation"] = json.dumps(tool["documentation"])

    if model_name == "bm25" or model_name == "BM25":
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

        print("Encoding tool embeddings...")
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

    qrels = {}
    for item in queries:
        qrels[item["id"]] = {str(x["id"]): int(x["relevance"]) for x in item["labels"]}

    collection = trec_eval(qrels=qrels, results=results)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        configs = {
            "model_name": model_name,
            "batch_size": batch_size,
            "top_k": top_k,
            "queries": queries,
            "tools": tools,
        }
        write_data(configs, os.path.join(save_dir, "configs.json"))
        write_data(results, os.path.join(save_dir, "details.json"))
        write_data(collection, os.path.join(save_dir, "results.json"))
    return {
        "summary": collection,
        "details": results,
    }


def eval_all_retrievers(
    save_dir: str,
    queries: List[Dict],
    tools: List[Dict],
    top_k: int = 100,
    redo: bool = False,
    additional_retrievers: List[Tuple[str, int]] = [],
) -> Dict[str, Dict]:
    total_summary = {}
    if additional_retrievers:
        retrievers_to_use = USE_RETRIEVER_MODELS_AND_BATCH_SIZE + additional_retrievers
        retrievers_to_use = list(set(retrievers_to_use))
    else:
        retrievers_to_use = USE_RETRIEVER_MODELS_AND_BATCH_SIZE
    for model_name, batch_size in retrievers_to_use:
        print(colored(f"=== Evaluating {model_name} ===", "cyan"))
        model_save_dir = os.path.join(
            save_dir, model_name.split("/")[-1] + "__" + str(batch_size)
        )
        if not redo and os.path.exists(os.path.join(model_save_dir, "results.json")):
            print(
                colored(
                    f"=== Skip {model_name} as results already exist in {model_save_dir} ===",
                    "yellow",
                )
            )
            with open(os.path.join(model_save_dir, "results.json"), "r") as f:
                total_summary[model_name] = json.load(f)
            continue
        os.makedirs(model_save_dir, exist_ok=True)
        ret = eval_retrieval(
            queries=queries,
            tools=tools,
            model_name=model_name,
            batch_size=batch_size,
            top_k=top_k,
            save_dir=model_save_dir,
        )
        total_summary[model_name] = ret["summary"]
        print(colored(f"=== Finished {model_name} ===", "green"))
    write_data(
        total_summary,
        os.path.join(save_dir, "all_retrieval_results.json"),
    )
    return total_summary


def write_data(data, filename, indent=4) -> None:
    if filename.endswith(".json"):
        json.dump(data, open(filename, "w"), indent=indent, ensure_ascii=False)
    elif filename.endswith(".jsonl"):
        with open(filename, "w") as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    elif filename.endswith(".txt"):
        with open(filename, "w") as f:
            for line in data:
                f.write(str(line) + "\n")
    elif filename.endswith(".pkl"):
        pickle.dump(data, open(filename, "wb"))
    elif filename.endswith(".yaml") or filename.endswith(".yml"):
        yaml.dump(data, open(filename, "w"), indent=indent, allow_unicode=True)
    else:
        raise ValueError("No suitable function to write data")
    print(f"=== Save {len(data)} data to {filename} ===")


def load_llm_client(
    model_name,
    server_url=None,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    max_tokens=2048,
):
    if model_name.startswith("gpt"):
        return LlmFactory.create(
            provider_name="openai",
            model=model_name,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
    else:
        # vllm
        assert server_url is not None, "Please provide vLLM server URL"
        return LlmFactory.create(
            provider_name="vllm",
            model=model_name,
            vllm_base_url=server_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )


def get_llm_name(llm: str) -> str:
    if "/" not in llm:
        return llm
    else:
        return llm.split("/")[-1]


def get_origin_field_name(field, dataset):
    if field == "description":
        return "description"
    elif field == "parameters":
        mapping = {
            "apibank": "parameters",
            "apigen": "parameters",
            "gorilla-huggingface": "api_arguments",
            "toolink": "func_description",
            "toolbench": "required_parameters",
        }
        return mapping.get(dataset, "parameters")
    elif field == "response":
        mapping = {
            "apibank": "responses",
            "toolbench": "template_response",
        }
        return mapping.get(dataset, "response")
    elif field == "examples":
        mapping = {
            "gorilla-huggingface": "example_code",
        }
        return mapping.get(dataset, "examples")

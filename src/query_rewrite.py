import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s
import faiss
import numpy as np
import yaml
from dotenv import load_dotenv
from src.utils import (
    USE_RETRIEVER_MODELS_AND_BATCH_SIZE,
    get_augment_tools_file,
    get_rewrite_queries_file,
    load_llm_client,
    load_local_data,
    write_data,
)
from termcolor import colored
from toolret.eval import RetModel
from tqdm import tqdm

load_dotenv()


# BM25 for description retrieval
def retrieve_description(
    queries: List[Dict],
    tools: List[Dict],
    model_name: str,
    batch_size: int = None,
    top_k: int = 100,
) -> Dict[str, List[Tuple[str, float]]]:
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
            results[item["id"]] = hits[0]
        return results

    else:
        if batch_size is None:
            for model, bs in USE_RETRIEVER_MODELS_AND_BATCH_SIZE.items():
                if (
                    model == model_name
                    or model.endswith(model_name)
                    or model.startswith(model_name)
                ):
                    batch_size = bs
                    break
        assert batch_size is not None, "Please specify batch_size for the retriever."
        model = RetModel(
            model_name,
        )
        tool_embeddings = model.encode_tools(tools, batch_size)
        tool_embeddings = np.asarray(tool_embeddings, dtype=np.float32)
        dim = tool_embeddings.shape[1]
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(tool_embeddings)

        query_embeddings = model.encode_queries(queries, batch_size)
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

        distance, rank = index.search(query_embeddings, top_k)

        results = {}
        for item, rk in zip(queries, rank):
            results[item["id"]] = []
            for i in range(len(rk)):
                tool_id = tools[rk[i]]["id"]
                results[item["id"]].append(tool_id)
        return results


def query_rewriting(
    item,
    llm_client,
    rewrite_prompt_template: Dict[str, str],
    description_retrieval_results: Dict[str, List[Tuple[str, float]]],
    tool_id_to_description: Dict[str, str],
    augment_tools: Dict[str, Dict],
) -> Tuple[Dict, bool]:
    messages = []
    for role, content in rewrite_prompt_template.items():
        retrieved_tool_descriptions = []
        for tid in range(len(description_retrieval_results[item["id"]])):
            tool_id = description_retrieval_results[item["id"]][tid]
            retrieved_tool_descriptions.append(
                f"Tool {tid}: {tool_id_to_description[tool_id]}"
            )
        item["retrieved_tool_descriptions"] = retrieved_tool_descriptions
        content = content.replace(
            "{retrieved_tool_descriptions}", "\n".join(retrieved_tool_descriptions)
        )
        content = content.replace("{user_query}", item["query"])
        view_tool = augment_tools[description_retrieval_results[item["id"]][0]][
            "augmented"
        ]
        content = content.replace("{tool_view}", json.dumps(view_tool))
        item["tool_view"] = view_tool
        messages.append({"role": role, "content": content})
    item["rewritten_messages"] = messages

    try_times = 5
    for _ in range(try_times):
        response = llm_client.generate_response(messages=messages)
        try:
            response_json = json.loads(response)
            if (
                "tool_needs" not in response_json
                or "extracted_arguments" not in response_json
                or len(response_json["tool_needs"]) == 0
            ):
                print("Invalid response format or empty tool_needs. Retrying...")
                continue
            item["rewritten_query"] = response_json
            return item, True
        except Exception as e:
            print(f"Failed to parse LLM response as JSON: {e}. Retrying...")
    raise RuntimeError("Failed to get valid response from LLM after multiple attempts.")


def main(args):
    llm_client = load_llm_client(args.llm, args.llm_server_url)
    queries, origin_tools = load_local_data(args.dataset)
    augment_tool_file = get_augment_tools_file(
        dataset_name=args.dataset,
        augment_llm=args.llm,
    )
    load_augment_tools = json.load(open(augment_tool_file, "r"))
    augment_tools = {tool["id"]: tool for tool in load_augment_tools}
    assert args.rewrite_prompt_file.endswith(
        ".yaml"
    ) or args.rewrite_prompt_file.endswith(".yml"), "Prompt file must be a yaml file."
    with open(args.rewrite_prompt_file, "r") as f:
        rewrite_prompt_template = yaml.safe_load(f)
    output_file = get_rewrite_queries_file(
        dataset_name=args.dataset,
        augment_and_rewrite_llm=args.llm,
        base_retriever=args.retriever,
        top_k=args.top_k,
        prompt_file=Path(os.path.basename(args.rewrite_prompt_file)).stem,
    )
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    if not args.redo and os.path.exists(output_file):
        print(
            colored(
                f"Rewritten queries already exist at {output_file}. Skipping rewriting.",
                "yellow",
            )
        )
        return json.load(open(output_file, "r")), load_augment_tools

    description_tools = deepcopy(origin_tools)
    for tool in description_tools:
        tool["documentation"] = augment_tools[tool["id"]]["augmented"]["description"]
    tool_id_to_description = {
        tool["id"]: tool["documentation"] for tool in description_tools
    }

    # retrieval top k
    description_retrieval_results = retrieve_description(
        queries=queries,
        tools=description_tools,
        model_name=args.retriever,
        batch_size=None,
        top_k=args.top_k,
    )

    # query rewriting
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_item = [
            executor.submit(
                query_rewriting,
                item,
                llm_client,
                rewrite_prompt_template,
                description_retrieval_results,
                tool_id_to_description,
                augment_tools,
            )
            for item in queries
        ]
        rewritten_queries = []
        failed_count = 0
        for future in tqdm(
            as_completed(future_to_item),
            total=len(future_to_item),
            desc="Rewriting queries",
        ):
            item, success = future.result()
            rewritten_queries.append(item)
            if not success:
                failed_count += 1
    rewritten_queries = sorted(
        rewritten_queries, key=lambda x: int(x["id"].split("_")[-1])
    )
    if failed_count == 0:
        print(colored("All queries rewritten successfully!", "green"))
    else:
        print(
            colored(
                f"Failed rewritings: {failed_count} / {len(rewritten_queries)}", "red"
            )
        )
    print(f"Save rewritten queries to {output_dir}")
    write_data(rewritten_queries, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--rewrite_prompt_file",
        type=str,
        default=None,
        help="Path to the prompt template file for query rewriting.",
    )
    parser.add_argument(
        "--retriever", type=str, default="bm25", help="Retriever model name."
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for query rewritten.",
    )
    parser.add_argument(
        "--llm_server_url", type=str, default=None, help="LLM server URL if applicable."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top k retrieval results to consider in query rewritten.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Whether to redo the rewriting even if the output file exists.",
    )
    args = parser.parse_args()
    if args.rewrite_prompt_file is None:
        args.rewrite_prompt_file = "./prompts/query_rewrite.yaml"
    main(args)

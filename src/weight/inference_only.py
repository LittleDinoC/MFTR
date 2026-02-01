import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from src.utils import USE_RETRIEVER_MODELS_AND_BATCH_SIZE, load_local_data, write_data
from src.weight.model import ToolRanker
from src.weight.utils import (
    calc_field_scores,
    eval_test_results,
    load_train_test_and_tools,
)
from tqdm import tqdm

load_dotenv()


def eval_on_test_set(
    test_queries: List[Dict],
    tools: List[Dict],
    field_scores: Dict,
    model: ToolRanker,
    max_params: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    save_top_k: int = 20,  # 默认保留前20个
) -> Tuple[Dict, Dict]:
    model.eval()
    preds = {}

    all_samples = []
    sample_indices = []

    for query in test_queries:
        for tool in tools:
            desc_score = field_scores["description"][query["id"]][tool["id"]]
            param_info_list = field_scores["parameters"][query["id"]][tool["id"]]
            resp_score = field_scores["response"][query["id"]][tool["id"]]
            example_score = field_scores["examples"][query["id"]][tool["id"]]
            avg_param_score = (
                np.mean([p["similarity"] for p in param_info_list])
                if param_info_list
                else 0.0
            )

            param_scores = []
            is_required = []
            for i in range(max_params):
                if i < len(param_info_list):
                    param_scores.append(param_info_list[i]["similarity"])
                    is_required.append(1.0 if param_info_list[i]["required"] else 0.0)
                else:
                    param_scores.append(100)
                    is_required.append(0.0)

            all_samples.append(
                {
                    "base_features": [
                        desc_score,
                        avg_param_score,
                        resp_score,
                        example_score,
                    ],
                    "param_scores": param_scores,
                    "is_required": is_required,
                }
            )
            sample_indices.append((query["id"], tool["id"]))

    all_scores = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(all_samples), batch_size), desc="Evaluating on test set"
        ):
            batch_samples = all_samples[i : i + batch_size]

            base_features = torch.tensor(
                [s["base_features"] for s in batch_samples], dtype=torch.float32
            ).to(device)
            param_scores = torch.tensor(
                [s["param_scores"] for s in batch_samples], dtype=torch.float32
            ).to(device)
            is_required = torch.tensor(
                [s["is_required"] for s in batch_samples], dtype=torch.float32
            ).to(device)

            scores, _, _, _ = model(base_features, param_scores, is_required)
            all_scores.extend(scores.squeeze(-1).cpu().tolist())

    for (query_id, tool_id), score in zip(sample_indices, all_scores):
        if query_id not in preds:
            preds[query_id] = {}
        preds[query_id][tool_id] = score

    for query_id in preds:
        tmp = sorted(preds[query_id].items(), key=lambda x: x[1], reverse=True)
        preds[query_id] = dict(tmp[:save_top_k])

    summary = eval_test_results(test_queries, preds)
    return preds, summary


def load_model_from_params(
    learned_params: Dict, max_params: int, device: str
) -> ToolRanker:
    model = ToolRanker(
        num_base_features=4,  # description score, avg_param_score, response score, example score
        soft_penalty_config={
            "init_tau": learned_params["tau"],
            "init_w_req": learned_params["w_req"],
            "init_w_opt": learned_params["w_opt"],
            "alpha": learned_params["alpha"],
        },
    ).to(device)

    with torch.no_grad():
        model.base_score_layer.weight.copy_(
            torch.tensor(learned_params["base_weight"], dtype=torch.float32)
        )
        model.base_score_layer.bias.copy_(
            torch.tensor(learned_params["base_bias"], dtype=torch.float32)
        )

        model.soft_penalty.raw_tau.copy_(
            torch.tensor(learned_params["tau"], dtype=torch.float32)
        )
        w_req_val = learned_params["w_req"]
        raw_w_req = np.log(np.exp(w_req_val) - 1) if w_req_val > 0 else 0.0
        model.soft_penalty.raw_w_req.copy_(torch.tensor(raw_w_req, dtype=torch.float32))

        w_opt_val = learned_params["w_opt"]
        raw_w_opt = np.log(np.exp(w_opt_val) - 1) if w_opt_val > 0 else 0.0
        model.soft_penalty.raw_w_opt.copy_(torch.tensor(raw_w_opt, dtype=torch.float32))

    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load data
    _data = load_train_test_and_tools(
        args.dataset,
        fold=args.fold_id,
        hard_negatives_from=args.hard_negatives_from,
    )
    train_queries = _data["train"]
    test_queries = _data["test"]
    tools = _data["tools"]

    # calc field scores
    total_queries = train_queries + test_queries
    field_scores = calc_field_scores(
        dataset=args.dataset,
        queries=total_queries,
        tools=tools,
        model_name=args.retriever,
        batch_size=args.retriever_batch_size,
        redo=False,
    )

    # calc max params
    max_params = 0
    for tool in tools:
        num_params = len(tool["augmented"]["parameters"])
        max_params = max(max_params, num_params)
    print(f"Max params: {max_params}")

    # load learned params
    learned_params_path = os.path.join(
        "weight",
        args.dataset,
        f"{args.retriever.split('/')[-1]}__{args.retriever_batch_size}",
        f"fold_{args.fold_id}_learned_params.json",
    )
    print(f"Loading learned params from: {learned_params_path}")
    with open(learned_params_path, "r") as f:
        learned_params = json.load(f)

    print("Learned params:")
    print(json.dumps(learned_params, indent=4))

    # load model
    model = load_model_from_params(learned_params, max_params, device)

    # eval on test set
    preds, summary = eval_on_test_set(
        test_queries=test_queries,
        tools=tools,
        field_scores=field_scores,
        model=model,
        max_params=max_params,
        device=device,
        batch_size=args.inference_batch_size,
    )

    # save results
    # write_data(preds, os.path.join(output_dir, "test_details.json"))
    # write_data(summary, os.path.join(output_dir, "test_summary.json"))

    print(f"===== Fold {args.fold_id} =====")
    print(summary)

    return preds, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use for testing.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=True,  # "ALL" means run all retrievers
        help="The retriever used for generating features.",
    )
    parser.add_argument(
        "--retriever_batch_size",
        type=int,
        default=None,
        help="The batch size used for feature generation.",
    )
    parser.add_argument(
        "--hard_negatives_from",
        type=str,
        default="bm25_origin",
        help="The source of hard negatives.",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )

    args = parser.parse_args()

    total_queries, _ = load_local_data(args.dataset)

    retriever_and_bsz = []
    if args.retriever != "ALL" and args.retriever_batch_size is None:
        for model, bs in USE_RETRIEVER_MODELS_AND_BATCH_SIZE:
            if args.retriever == model or model.endswith(args.retriever):
                args.retriever = model
                args.retriever_batch_size = bs
                break
        retriever_and_bsz.append((args.retriever, args.retriever_batch_size))
    else:
        retriever_and_bsz = USE_RETRIEVER_MODELS_AND_BATCH_SIZE
    print(retriever_and_bsz)

    for retriever_name, retriever_bsz in retriever_and_bsz:
        args.retriever = retriever_name
        args.retriever_batch_size = retriever_bsz

        output_dir = os.path.join(
            "output",
            "reproduction",
            args.dataset,
            f"{args.retriever.split('/')[-1]}__{args.retriever_batch_size}",
        )
        os.makedirs(output_dir, exist_ok=True)

        total_preds = {}
        for fold in range(5):
            args.fold_id = fold
            preds, _ = main(args)
            total_preds.update(preds)
        write_data(total_preds, os.path.join(output_dir, "details.json"))
        print(
            f"All folds test details saved to: {os.path.join(output_dir, 'details.json')}"
        )
        summary = eval_test_results(total_queries, total_preds)
        write_data(summary, os.path.join(output_dir, "results.json"))
        print(
            f"All folds test summary saved to: {os.path.join(output_dir, 'results.json')}"
        )
        print("===== Overall Summary =====")
        print(summary)

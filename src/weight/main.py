import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from src.utils import USE_RETRIEVER_MODELS_AND_BATCH_SIZE, write_data
from src.weight.model import ToolRanker
from src.weight.utils import (
    calc_field_scores,
    eval_test_results,
    load_train_test_and_tools,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
load_dotenv()


class ToolRankingDataset(Dataset):
    def __init__(
        self,
        queries: List[Dict],
        tools: List[Dict],
        field_scores: Dict,
        hard_negatives: Dict,
        hard_negatives_cnt: int,
        max_params: int,
    ):
        self.field_scores = field_scores
        self.samples = []
        self.max_params = max_params
        tool_id_to_tool = {tool["id"]: tool for tool in tools}

        for query in queries:
            qid = query["id"]
            gold_tool_ids = set(
                label["id"] for label in query["labels"] if label["relevance"] == 1
            )

            # 负样本
            if hard_negatives_cnt == -1:
                all_neg_ids = set(tool_id_to_tool.keys()) - gold_tool_ids
            else:
                hard_negs = hard_negatives.get(query["id"], [])[:hard_negatives_cnt]
                all_neg_ids = set(tool_idx[0] for tool_idx in hard_negs)
            assert len(gold_tool_ids & all_neg_ids) == 0, (
                "Gold and negative sets overlap!"
            )

            # 构建 pairs
            for g_id in gold_tool_ids:
                for n_id in all_neg_ids:
                    self.samples.append(
                        {
                            "query_id": qid,
                            "pos_tool": tool_id_to_tool[g_id],
                            "neg_tool": tool_id_to_tool[n_id],
                        }
                    )

    def _prepare_features(self, query_id: str, tool: Dict) -> Dict:
        desc_score = self.field_scores["description"][query_id][tool["id"]]
        param_info_list = self.field_scores["parameters"][query_id][tool["id"]]
        resp_score = self.field_scores["response"][query_id][tool["id"]]
        example_score = self.field_scores["examples"][query_id][tool["id"]]
        avg_param_score = (
            np.mean([p["similarity"] for p in param_info_list])
            if param_info_list
            else 0.0
        )

        # Pad or truncate to max_params
        param_scores = []
        is_required = []
        for i in range(self.max_params):
            if i < len(param_info_list):
                param_scores.append(param_info_list[i]["similarity"])
                is_required.append(1.0 if param_info_list[i]["required"] else 0.0)
            else:
                param_scores.append(
                    100
                )  # padding, treated as perfect match, with no penalty
                is_required.append(0.0)  # padding treated as not required

        return {
            "base_features": torch.tensor(
                [
                    desc_score,
                    avg_param_score,
                    resp_score,
                    example_score,
                ],
                dtype=torch.float32,
            ),
            "param_scores": torch.tensor(param_scores, dtype=torch.float32),
            "is_required": torch.tensor(is_required, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair = self.samples[idx]
        pos_feat = self._prepare_features(pair["query_id"], pair["pos_tool"])
        neg_feat = self._prepare_features(pair["query_id"], pair["neg_tool"])
        return pos_feat, neg_feat


class RankNetLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(RankNetLoss, self).__init__()
        self.sigma = sigma

    def forward(self, pos_scores, neg_scores):
        diff = pos_scores - neg_scores
        loss = torch.log(1 + torch.exp(-self.sigma * diff))
        return loss.mean()


def train_soft_penalty_model(
    train_queries: List[Dict],
    tools: List[Dict],
    field_scores: Dict,
    hard_negatives: Dict,
    hard_negatives_cnt: int,
    max_params: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_epochs: int = 200,
    lr: float = 0.01,
    batch_size: int = 64,
) -> Tuple[ToolRanker, Dict]:
    dataset = ToolRankingDataset(
        queries=train_queries,
        tools=tools,
        field_scores=field_scores,
        hard_negatives=hard_negatives,
        hard_negatives_cnt=hard_negatives_cnt,
        max_params=max_params,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = ToolRanker(
        num_base_features=len(
            field_scores
        ),  # description score, avg_param_score, response score, example score
        soft_penalty_config={
            "init_tau": 0.3,
            "init_w_req": 0.2,
            "init_w_opt": 0.1,
            "alpha": 15.0,
        },
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RankNetLoss(sigma=1.0)
    training_details = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for pos_batch, neg_batch in dataloader:
            pos_scores, tau, w_req, w_opt = model(
                pos_batch["base_features"].to(device),
                pos_batch["param_scores"].to(device),
                pos_batch["is_required"].to(device),
            )
            neg_scores, _, _, _ = model(
                neg_batch["base_features"].to(device),
                neg_batch["param_scores"].to(device),
                neg_batch["is_required"].to(device),
            )
            optimizer.zero_grad()
            loss = criterion(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, "
                f"Base weight: {model.base_score_layer.weight.detach().cpu().numpy()}, "
                f"tau: {tau.item():.4f}, w_req: {w_req.item():.4f}, w_opt: {w_opt.item():.4f}"
            )
            training_details.append(
                {
                    "epoch": epoch + 1,
                    "loss": total_loss / len(dataloader),
                    "base_weight": model.base_score_layer.weight.detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "tau": tau.item(),
                    "w_req": w_req.item(),
                    "w_opt": w_opt.item(),
                }
            )

    learned_params = {
        "base_weight": model.base_score_layer.weight.detach().cpu().numpy().tolist(),
        "base_bias": model.base_score_layer.bias.detach().cpu().numpy().tolist(),
        "tau": torch.clamp(model.soft_penalty.raw_tau, 0.0, 1.0).detach().cpu().item(),
        "w_req": torch.nn.functional.softplus(model.soft_penalty.raw_w_req)
        .detach()
        .cpu()
        .item(),
        "w_opt": torch.nn.functional.softplus(model.soft_penalty.raw_w_opt)
        .detach()
        .cpu()
        .item(),
        "alpha": model.soft_penalty.alpha,
    }

    print("===== Learned Soft Penalty Parameters =====")
    print(f"Base weight: {learned_params['base_weight']}")
    print(f"Base bias: {learned_params['base_bias']}")
    print(f"tau: {learned_params['tau']:.4f}")
    print(f"w_req: {learned_params['w_req']:.4f}")
    print(f"w_opt: {learned_params['w_opt']:.4f}")

    return model, learned_params, training_details


def eval_on_test_set(
    test_queries: List[Dict],
    tools: List[Dict],
    field_scores: Dict,
    model: ToolRanker,
    max_params: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    save_top_k: int = 20,
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

    # 排序
    for query_id in preds:
        tmp = sorted(preds[query_id].items(), key=lambda x: x[1], reverse=True)
        preds[query_id] = dict(tmp[:save_top_k])  # 只保留前20个

    summary = eval_test_results(test_queries, preds)
    return preds, summary


def main(args):
    output_dir = os.path.join(
        "output",
        "weighted_field_retrieval",
        f"epochs_{args.num_epochs}__lr_{args.lr}__bs_{args.training_batch_size}__{args.hard_negatives_from}__neg_{args.hard_negatives_cnt}",
        args.dataset,
        f"{args.retriever.split('/')[-1]}__{args.retriever_batch_size}",
    )
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    folds = [args.fold_id] if args.fold_id >= 0 else [0, 1, 2, 3, 4]
    all_fold_details = {}
    all_fold_summary = {}
    for fold_id in folds:
        _data = load_train_test_and_tools(
            args.dataset,
            fold=fold_id,
            hard_negatives_from=args.hard_negatives_from,
        )
        train_queries = _data["train"]
        test_queries = _data["test"]
        tools = _data["tools"]
        hard_negatives = _data["hard_negatives"]

        fold_dir = os.path.join(output_dir, f"fold_{fold_id}")

        if (
            not args.redo
            and os.path.exists(os.path.join(fold_dir, "test_summary.json"))
            and os.path.exists(os.path.join(fold_dir, "test_details.json"))
        ):
            details = json.load(open(os.path.join(fold_dir, "test_details.json")))
            summary = json.load(open(os.path.join(fold_dir, "test_summary.json")))
        else:
            total_queries = train_queries + test_queries
            field_scores = calc_field_scores(
                dataset=args.dataset,
                queries=total_queries,
                tools=tools,
                model_name=args.retriever,
                batch_size=args.retriever_batch_size,
                redo=args.redo,
            )

            # 获取所有工具中最大的参数个数
            max_params = 0
            for tool in tools:
                num_params = len(tool["augmented"]["parameters"])
                max_params = max(max_params, num_params)
            print(f"Max params: {max_params}")

            model, learned_params, training_details = train_soft_penalty_model(
                train_queries=train_queries,
                tools=tools,
                field_scores=field_scores,
                hard_negatives=hard_negatives,
                hard_negatives_cnt=args.hard_negatives_cnt,
                max_params=max_params,
                device=device,
                num_epochs=args.num_epochs,
                lr=args.lr,
                batch_size=args.training_batch_size,
            )

            os.makedirs(fold_dir, exist_ok=True)
            write_data(learned_params, os.path.join(fold_dir, "learned_params.json"))
            write_data(
                training_details, os.path.join(fold_dir, "training_details.json")
            )
            torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))

            details, summary = eval_on_test_set(
                test_queries=test_queries,
                tools=tools,
                field_scores=field_scores,
                model=model,
                max_params=max_params,
                device=device,
            )

            write_data(details, os.path.join(fold_dir, "test_details.json"))
            write_data(summary, os.path.join(fold_dir, "test_summary.json"))

        print(f"===== Fold {fold_id} Evaluation Summary =====")
        print(summary)
        all_fold_details.update(details)
        all_fold_summary[fold_id] = summary

    if args.fold_id == -1:
        print("\n\nAll:")
        _data = load_train_test_and_tools(
            args.dataset,
            fold=0,
            hard_negatives_from=args.hard_negatives_from,
        )
        all_queries = _data["train"] + _data["test"]
        each_item_summary = eval_test_results(all_queries, all_fold_details)
        merge_test_dir = os.path.join(output_dir, "merge_test")
        os.makedirs(merge_test_dir, exist_ok=True)
        write_data(all_fold_details, os.path.join(merge_test_dir, "details.json"))
        print(each_item_summary)
        write_data(each_item_summary, os.path.join(merge_test_dir, "results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use for training and testing.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=True,  # "ALL" means run all retrievers
        help="The retriever to use for generating features.",
    )
    parser.add_argument(
        "--retriever_batch_size",
        type=int,
        default=None,
        help="The batch size to use for feature generation.",
    )
    parser.add_argument(
        "--fold_id",
        type=int,
        default=-1,  # use all folds
        help="The fold ID for cross-validation.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Whether to redo the feature calculation.",
    )
    parser.add_argument(
        "--hard_negatives_from",
        type=str,
        default="bm25_origin",
        help="The source of hard negatives.",
    )
    parser.add_argument(
        "--hard_negatives_cnt",
        type=int,
        default=64,
        help="The number of hard negatives to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=256,
        help="Batch size for training.",
    )

    args = parser.parse_args()
    if args.retriever == "ALL":
        assert args.retriever_batch_size is None, (
            "When using ALL retrievers, batch_size must be None."
        )
        assert args.fold_id == -1, (
            "When using ALL retrievers, fold_id must be -1 (all folds)."
        )
        for model, bs in USE_RETRIEVER_MODELS_AND_BATCH_SIZE:
            args.retriever = model
            args.retriever_batch_size = bs
            main(args)

    else:
        if args.retriever_batch_size is None:
            for model, bs in USE_RETRIEVER_MODELS_AND_BATCH_SIZE:
                if args.retriever == model or model.endswith(args.retriever):
                    args.retriever = model
                    args.retriever_batch_size = bs
        main(args)

import argparse
import json
import os

from datasets import load_dataset
from src.utils import write_data


def save_dataset(save_dir):
    dataset_list = [
        "apibank",
        "apigen",
        "gorilla-huggingface",
        "toolink",
        "toolbench",
    ]

    def dataset_name_to_tool_prefix(dataset_name) -> str:
        if dataset_name == "gorilla-huggingface":
            return "gorilla_huggingface"
        return dataset_name

    for dataset_name in dataset_list:
        save_path = os.path.join(save_dir, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        print(f"Processing dataset: {dataset_name}, save to {save_path}")
        query_dataset = load_dataset(
            "mangopy/ToolRet-Queries", dataset_name, split="queries"
        )
        queries = []
        for item in query_dataset.to_list():
            item["labels"] = json.loads(item["labels"])
            queries.append(item)
        print(f"= Loaded {len(queries)} queries for dataset {dataset_name}")
        write_data(queries, os.path.join(save_path, "queries.json"))

        tool_prefix = dataset_name_to_tool_prefix(dataset_name)
        total_tools = {}
        for name in ["code", "customized", "web"]:
            total_tools[name] = load_dataset(
                "mangopy/ToolRet-Tools", name, split="tools"
            )
        tools = []
        for name, data in total_tools.items():
            for tool in data.to_list():
                if tool["id"].startswith(tool_prefix):
                    tool["type"] = name
                    tool["documentation"] = json.loads(tool["documentation"])
                    tools.append(tool)
        print(f"= Loaded {len(tools)} tools for dataset {dataset_name}\n")
        write_data(tools, os.path.join(save_path, "tools.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dataset/from_toolret/",
        help="Directory to save the processed datasets.",
    )
    args = parser.parse_args()
    save_dataset(args.save_dir)

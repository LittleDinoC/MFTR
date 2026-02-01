import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv
from src.utils import (
    get_augment_tools_file,
    get_llm_name,
    load_llm_client,
    load_local_data,
    write_data,
)
from termcolor import colored
from tqdm import tqdm

load_dotenv()

AUGMENT_FIELDS = ["description", "parameters", "response", "examples"]


def load_prompt_file(prompt_file: str) -> Dict[str, str]:
    prompt_file = str(prompt_file)
    assert prompt_file.endswith(".yaml") or prompt_file.endswith(".yml"), (
        "Prompt file must be a YAML file."
    )
    with open(prompt_file, "r") as fin:
        prompts = yaml.safe_load(fin)
    return prompts


def fill_template(prompt_template: Dict[str, str], **kwargs) -> List[Dict[str, str]]:
    messages = []
    for role, content in prompt_template.items():
        for fill_key, fill_value in kwargs.items():
            content = content.replace(f"{{{fill_key}}}", fill_value)
        messages.append({"role": role, "content": content})
    return messages


def augment_tool(
    tool: Dict, augment_prompts: Dict[str, str], llm_client, field_name: str
) -> Dict:
    origin_doc = tool["documentation"]
    messages = fill_template(augment_prompts, origin_doc=json.dumps(origin_doc))
    for _ in range(10):
        try:
            response = llm_client.generate_response(messages=messages)
            if field_name in ["parameters", "examples"]:
                try:
                    # 针对 parameters 字段，尝试将返回值解析为 JSON 格式
                    response_json = json.loads(response)
                    response = response_json
                except Exception:
                    print(
                        colored(
                            f"Warning: Failed to parse LLM response as JSON for field '{field_name}'. Retry.",
                            "yellow",
                        )
                    )
                    continue
            break
        except Exception as e:
            print(colored(f"Error during LLM generation: {e}. Retrying...", "red"))
    else:
        raise RuntimeError("Failed to get response from LLM after 10 attempts.")

    tool["augment"] = {
        "field": field_name,
        "origin_value": origin_doc.get(field_name, None),
        "new_value": response,
    }
    tool["documentation"][field_name] = response

    return tool


def augment_field(
    args, origin_tools: List[Dict], field_name: str, llm_client
) -> List[Dict]:
    """对单个字段进行增强"""
    prompt_file = getattr(args, f"{field_name}_prompt_file", None)
    if prompt_file is None:
        prompt_file = f"./prompts/{field_name}.yaml"

    assert os.path.exists(prompt_file), (
        f"Prompt file {prompt_file} for field {field_name} does not exist."
    )

    augment_prompts = load_prompt_file(prompt_file)
    prompt_name = os.path.basename(prompt_file).replace(".yaml", "").replace(".yml", "")

    output_dir = os.path.join(
        args.output_dir,
        args.dataset,
        field_name,
        prompt_name,
        get_llm_name(args.llm),
    )
    os.makedirs(output_dir, exist_ok=True)

    write_data(augment_prompts, os.path.join(output_dir, "augment_prompt.yaml"))

    aug_tool_save_path = os.path.join(output_dir, "augmented_tools.json")

    if os.path.exists(aug_tool_save_path):
        print(colored(f"Loading existing augmented tools for {field_name}...", "cyan"))
        aug_tools = json.load(open(aug_tool_save_path, "r"))
    else:
        print(colored(f"Augmenting field: {field_name}...", "blue"))
        aug_tools = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_tool = [
                executor.submit(
                    augment_tool,
                    deepcopy(tool),
                    augment_prompts,
                    llm_client,
                    field_name,
                )
                for tool in origin_tools
            ]
            for future in tqdm(
                as_completed(future_to_tool),
                total=len(future_to_tool),
                desc=f"Augmenting {field_name}",
            ):
                aug_tools.append(future.result())

        aug_tools = sorted(aug_tools, key=lambda x: int(x["id"].split("_")[-1]))
        assert (
            len(aug_tools)
            == len(origin_tools)
            == len(set([x["id"] for x in aug_tools]))
        ), f"Augmented tools count mismatch for {field_name}"
        write_data(aug_tools, aug_tool_save_path)
        print(
            colored(
                f"Saved augmented tools for {field_name} to {aug_tool_save_path}",
                "green",
            )
        )

    return aug_tools


def merge_augmented_tools(
    origin_tools: List[Dict], augmented_results: Dict[str, List[Dict]]
) -> List[Dict]:
    """合并所有字段的增强结果"""
    merged_tools = deepcopy(origin_tools)

    for tool in merged_tools:
        tool["augmented"] = {}
        tool["origin_documentation"] = tool.pop("documentation", {})

    for field_name, aug_tools in augmented_results.items():
        if aug_tools is None:
            continue
        for merged_tool, aug_tool in zip(merged_tools, aug_tools):
            assert merged_tool["id"] == aug_tool["id"], (
                f"Tool IDs do not match: {merged_tool['id']} != {aug_tool['id']}"
            )
            merged_tool["augmented"][field_name] = aug_tool["documentation"][field_name]

    return merged_tools


def main(args):
    print(colored(f"Processing dataset: {args.dataset}", "blue"))

    llm_client = load_llm_client(args.llm, args.llm_server_url)
    queries, origin_tools = load_local_data(args.dataset)

    # augment all fields
    augmented_results = {}
    for field_name in AUGMENT_FIELDS:
        aug_tools = augment_field(args, origin_tools, field_name, llm_client)
        augmented_results[field_name] = aug_tools

    # check missing fields
    missing_fields = [f for f, v in augmented_results.items() if v is None]
    if missing_fields:
        print(
            colored(
                f"Error: Missing augmented results for fields: {missing_fields}",
                "red",
            )
        )
        raise ValueError("Missing augmented results.")

    # merge all fields
    merged_tools = merge_augmented_tools(origin_tools, augmented_results)

    # save merged results
    output_file = get_augment_tools_file(
        dataset_name=args.dataset,
        augment_llm=args.llm,
    )
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_data(merged_tools, output_file)
    print(
        colored(
            f"Merged augmented tools saved to {output_file}",
            "green",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment all fields and merge results for a dataset."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/augment_field/",
        help="Directory to save augmented data.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for augmentation.",
    )
    parser.add_argument(
        "--llm_server_url",
        type=str,
        default=None,
        help="LLM server URL if applicable.",
    )
    # 为每个字段指定不同的 prompt 文件
    parser.add_argument(
        "--description_prompt_file",
        type=str,
        default=None,
        help="Prompt file for description augmentation.",
    )
    parser.add_argument(
        "--parameters_prompt_file",
        type=str,
        default=None,
        help="Prompt file for parameters augmentation.",
    )
    parser.add_argument(
        "--response_prompt_file",
        type=str,
        default=None,
        help="Prompt file for response augmentation.",
    )
    parser.add_argument(
        "--examples_prompt_file",
        type=str,
        default=None,
        help="Prompt file for examples augmentation.",
    )
    args = parser.parse_args()
    main(args)

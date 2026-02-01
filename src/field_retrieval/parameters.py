import json
from typing import Dict, List

from src.field_retrieval.base import FieldRetrievalBase

PARAMETER_FORMAT_TYPES = [
    "description_only",
    "with_type",
    "with_default",
    "with_type_and_default",
    "full",
    "json",
]


def parameter_formatter(param_name: str, param: Dict, form: str) -> str:
    if form == "description_only":
        return f"{param_name}: {param['description']}"
    if form == "with_type":
        return f"{param_name} ({param['type']}): {param['description']}"
    if form == "with_default":
        default = param.get("default", "None")
        return f"{param_name} (default: {default}): {param['description']}"
    if form == "with_type_and_default":
        default = param.get("default", "None")
        return f"{param_name} (type: {param['type']}, default: {default}): {param['description']}"
    if form == "full":
        default = param.get("default", "None")
        s = f"{param_name} ({param['type']}, default: {default}): {param['description']}"
        example = param.get("example", None)
        if example is not None:
            s += f". Example: {example}"
        return s
    if form == "json":
        try:
            param_copy = param.copy()
            param_copy["name"] = param_name
            return json.dumps(param_copy)
        except Exception:
            return json.dumps(param)


def arguement_format(name, val) -> str:
    return f"{name}: {val}"


class ParametersRetrieval(FieldRetrievalBase):
    DEFAULT_PARAMETER_FORMAT = "json"

    def __init__(self, parameter_format: str = DEFAULT_PARAMETER_FORMAT, **kwargs):
        self.parameter_format = parameter_format
        assert self.parameter_format in PARAMETER_FORMAT_TYPES, (
            f"Invalid parameter_format: {self.parameter_format}"
        )
        super().__init__(**kwargs)

    def get_query_arg_id(self, query_id: str, arg_name: str):
        return f"{query_id}__arg_{arg_name}"

    def get_tool_param_id(self, tool_id: str, param_name: str):
        return f"{tool_id}__param_{param_name}"

    def preprocess_data(
        self,
        raw_queries: List[Dict],
        raw_tools: List[Dict],
    ) -> List[Dict]:
        tools = []
        for tool in raw_tools:
            for param_name, param_values in tool["augmented"]["parameters"].items():
                tools.append(
                    {
                        "id": self.get_tool_param_id(tool["id"], param_name),
                        "documentation": parameter_formatter(
                            param_name, param_values, self.parameter_format
                        ),
                    }
                )
        queries = []
        for query in raw_queries:
            for arg_name, arg_desc in query["rewritten_query"][
                "extracted_arguments"
            ].items():
                queries.append(
                    {
                        "id": self.get_query_arg_id(query["id"], arg_name),
                        "query": arguement_format(arg_name, arg_desc),
                    }
                )
        return queries, tools

    def retrieve(
        self,
        queries: List[Dict],
        tools: List[Dict],
    ) -> Dict[str, List[str]]:
        processed_queries, processed_tools = self.preprocess_data(queries, tools)
        processed_retrieval_results = self.retrieve_processed_data(
            processed_queries, processed_tools
        )
        retrieval_results = {}
        for query in queries:
            arg_ids = list(
                self.get_query_arg_id(query["id"], arg_name)
                for arg_name in query["rewritten_query"]["extracted_arguments"].keys()
            )
            tool_scores = {}
            for tool in tools:
                param_score = []
                for param_name, param_values in tool["augmented"]["parameters"].items():
                    best_similarity = 0.0
                    for arg_id in arg_ids:
                        ret_score = processed_retrieval_results[arg_id].get(
                            self.get_tool_param_id(tool["id"], param_name), 0.0
                        )
                        best_similarity = max(best_similarity, ret_score)
                    param_score.append(
                        {
                            "similarity": best_similarity,
                            "required": param_values.get("required", True),
                        }
                    )
                tool_scores[tool["id"]] = param_score
            retrieval_results[query["id"]] = tool_scores
        return {
            "processed_queries": processed_queries,
            "processed_tools": processed_tools,
            "retrieval_results": retrieval_results,
        }

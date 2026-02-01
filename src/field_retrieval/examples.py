from typing import Dict, List

from src.field_retrieval.base import FieldRetrievalBase


class MaxNeeds_ExamplesRetrieval(FieldRetrievalBase):
    def get_tool_need_id(self, query_id: str, need_id: int):
        return f"{query_id}__need_{need_id}"

    def preprocess_data(
        self,
        raw_queries: List[Dict],
        raw_tools: List[Dict],
    ) -> List[Dict]:
        tools = []
        for tool in raw_tools:
            if len(tool["augmented"]["examples"]) == 0:
                tools.append(
                    {
                        "id": tool["id"],
                        "documentation": "",
                    }
                )
                continue
            for eid, example in enumerate(tool["augmented"]["examples"]):
                # print(tool["id"], eid)
                tools.append(
                    {
                        "id": f"{tool['id']}__example_{eid}",
                        "documentation": example["user_intent"],
                    }
                )
        queries = []
        for query in raw_queries:
            for need_id, tool_need in enumerate(query["rewritten_query"]["tool_needs"]):
                queries.append(
                    {
                        "id": self.get_tool_need_id(query["id"], need_id),
                        "query": tool_need["user_intent"],
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
            # 每个 tool 的分数是所有 need 中的最大值
            need_indices = [
                self.get_tool_need_id(query["id"], need_id)
                for need_id in range(len(query["rewritten_query"]["tool_needs"]))
            ]
            tool_scores = {}
            for need_id in need_indices:
                need_retrievals = processed_retrieval_results[need_id]
                for tool_id, score in need_retrievals.items():
                    tool_id = tool_id.split("__example_")[0]
                    if tool_id not in tool_scores:
                        tool_scores[tool_id] = score
                    else:
                        tool_scores[tool_id] = max(tool_scores[tool_id], score)
            sorted_items = sorted(
                tool_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            retrieval_results[query["id"]] = {k: v for k, v in sorted_items}
        return {
            "processed_queries": processed_queries,
            "processed_tools": processed_tools,
            "retrieval_results": retrieval_results,
        }

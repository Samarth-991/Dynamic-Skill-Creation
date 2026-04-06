import sys
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Optional, List, Dict, Any
import json
from pathlib import Path

class AgentState(TypedDict):
    messages:           Annotated[list, add_messages]
    selected_skill:     Optional[str]
    skill_instructions: Optional[str]
    tool_results:       List[dict]
    final_response:     Optional[str]
    token_usage:        Dict  # cumulative {input, output, total} across all LLM calls


def extract_token_usage(response) -> Dict:
    """Pull input/output/total token counts from a Gemini LangChain response."""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    # langchain-google-genai stores usage in response_metadata
    meta = getattr(response, "response_metadata", {}) or {}
    # Gemini SDK key names
    usage_meta = meta.get("usage_metadata") or meta.get("token_counts") or {}
    if usage_meta:
        usage["input_tokens"]  = usage_meta.get("prompt_token_count")     or usage_meta.get("input_tokens",  0)
        usage["output_tokens"] = usage_meta.get("candidates_token_count") or usage_meta.get("output_tokens", 0)
        usage["total_tokens"]  = usage_meta.get("total_token_count")       or usage_meta.get("total_tokens",  0)
    # Fallback: langchain standard usage_metadata attribute
    if usage["total_tokens"] == 0 and hasattr(response, "usage_metadata"):
        um = response.usage_metadata or {}
        usage["input_tokens"]  = um.get("input_tokens",  0)
        usage["output_tokens"] = um.get("output_tokens", 0)
        usage["total_tokens"]  = um.get("total_tokens",  0)
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def merge_usage(a: Dict, b: Dict) -> Dict:
    """Add two token-usage dicts together."""
    return {
        "input_tokens":  a.get("input_tokens",  0) + b.get("input_tokens",  0),
        "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
        "total_tokens":  a.get("total_tokens",  0) + b.get("total_tokens",  0),
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tools"
    return "end"

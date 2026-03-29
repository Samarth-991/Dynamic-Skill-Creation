"""
skill_agent.py

LangGraph agent that replicates Claude's skill execution pipeline.

STEPS:
  STEP 1  — SKILL DISCOVERY   (system prompt injection from registry)
  STEP 2  — SKILL ROUTING     (LLM picks best skill from descriptions)
  STEP 3  — SKILL READING     (read_skill_instructions called first)
  STEP 4  — SKILL EXECUTION   (LLM follows SKILL.md workflow, calls tools)
  STEP 5  — RESPONSE GENERATION (clean Markdown, never raw objects)

LLM: provided by user at runtime (OpenAI, Azure, Groq, etc) — agent should be model-agnostic
Orchestration: LangGraph StateGraph
"""
import os
import sys
import re
import json
import importlib
from pathlib import Path
from typing import TypedDict, Annotated, Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

sys.path.append(str(Path(__file__).parent.parent))  # adjust as needed to find handler and utils
from agent.agent_graph import should_continue,AgentState,extract_token_usage,merge_usage
from handler.skills_registry import get_registry , format_skills_for_prompt,get_skill_instructions
from utils.agent_utils import extract_text_content
from handler.handler_chains import is_skill_query_chain

SKILL_DISPLAY_NAMES = {
    "browser-search": "Web Search 🌐",
    "opencv-image-reader": "Image Reader 🖼️",
    # Add more mappings as needed
}


@tool
def list_available_skills() -> str:
    """
    Return a clean, numbered Markdown list of every skill that is currently
    available. Call this when the user asks what skills or capabilities exist.
    Present the returned text directly — do not reformat or summarise it.
    """
    registry = get_registry()
    if not registry:
        return "No skills are currently loaded."

    lines = ["## 🧠 Available Skills\n"]
    for i, (name, skill) in enumerate(registry.items(), 1):
        display = SKILL_DISPLAY_NAMES.get(name, name.replace("-", " ").title())
        # First sentence of description only
        short_desc = skill["description"].split(". ")[0].rstrip(".")
        lines.append(f"### {i}. {display}")
        lines.append(f"{short_desc}.\n")

    lines.append("---")
    lines.append("_Simply describe what you need and I will automatically use the right skill._")
    return "\n".join(lines)

@tool
def web_page_scraper_tool(input_value: str) -> str:
    """Searches for and scrapes web pages to extract titles, headers, and main text content."""
    import sys
    import json
    from pathlib import Path

    scripts_dir = Path(__file__).parent / "skills" / "web-page-scraper" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import web_page_scraper
        result = web_page_scraper.run_web_page_scraper(input_value)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "error_type": type(e).__name__})
    finally:
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))

@tool
def read_skill_instructions(skill_name: str) -> str:
    """
    Read the full SKILL.md workflow instructions for a skill before executing it.
    MUST be called before any skill-specific tool so the LLM knows the workflow.

    Args:
        skill_name: Exact skill name, e.g. 'youtube-transcript'.
    Returns:
        Full SKILL.md body with workflow, patterns, error handling, and output format.
    """
    registry = get_registry()
    instructions = get_skill_instructions(registry, skill_name)
    if not instructions:
        return f"Skill '{skill_name}' not found. Available: {list(registry.keys())}"
    return instructions

@tool
def browser_search_tool(query: str) -> str:
    """Perform a web search and return a list of relevant URLs and snippets."""
    import sys
    import json
    from pathlib import Path

    scripts_dir = Path(__file__).parent / "skills" / "browser-search" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import browser_search
        result = browser_search.run_browser_search(query)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "error_type": type(e).__name__})
    finally:
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))

@tool
def opencv_image_reader_tool(input_value: str) -> str:
    """Read, load, or open an image using OpenCV in Python"""
    scripts_dir = Path(__file__).parent / "skills" / "opencv-image-reader" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import opencv_image_reader
        result = opencv_image_reader.run_opencv_image_reader(input_value)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except ImportError as e:
        return json.dumps({"error": str(e), "error_type": type(e).__name__})
    except Exception as e:
        return json.dumps({"error": str(e), "error_type": type(e).__name__})
    finally:
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))


TOOLS_LIST = [
    list_available_skills, web_page_scraper_tool, read_skill_instructions, opencv_image_reader_tool]
TOOLS    = list(TOOLS_LIST)
TOOL_MAP = {t.name: t for t in TOOLS}


def reload_tools():
    """
    Rebuild TOOLS and TOOL_MAP after create_skill.py has injected a new @tool stub.
    Called by app.py so the new skill is live in chat immediately.

    Handles both local (module key = 'skill_agent') and Streamlit Cloud
    (module key = full dotted path like 'implementing_skills_using_langchain.skill_agent')
    by searching all sys.modules values for this file's path.
    """
    global TOOLS, TOOL_MAP, AGENT_GRAPH

    this_file = Path(__file__).resolve()
    reloaded  = False

    # Search every loaded module for one whose __file__ matches ours
    for key, mod in list(sys.modules.items()):
      mod_file = getattr(mod, "__file__", None)
      if mod_file and Path(mod_file).resolve() == this_file:
          try:
            mod      = importlib.reload(mod)
            TOOLS    = mod.TOOLS_LIST
            TOOL_MAP = {t.name: t for t in TOOLS}
            reloaded = True
            print(f"[SkillAgent] Reloaded via key='{key}' — {len(TOOLS)} tools")
            break
          except Exception as e:
            print(f"[SkillAgent] Reload failed for key='{key}': {e}")

    if not reloaded:
        # Module not in sys.modules yet — just rebuild from current globals
        TOOLS    = list(TOOLS_LIST)
        TOOL_MAP = {t.name: t for t in TOOLS}
        print(f"[SkillAgent] Cold-build — {len(TOOLS)} tools")

    AGENT_GRAPH = build_graph()
    print(f"[SkillAgent] Tools active: {[t.name for t in TOOLS]}")

def build_system_prompt(registry: Optional[Dict] = None, executed_tools: Optional[List] = None) -> str:
    if registry is None:
        registry = get_registry()
    skills_block = format_skills_for_prompt(registry)

    # Build a "already done" block so Gemini knows not to repeat calls
    done_block = ""
    if executed_tools:
        done_lines = []
        for tr in executed_tools:
            done_lines.append(f"  - {tr['tool']}({json.dumps(tr['args'])[:80]}) → DONE")
        done_block = (
            "\n## ✅ Tools Already Executed This Turn (DO NOT call again)\n"
            + "\n".join(done_lines)
            + "\n\nAll results are already in the conversation above. "
            "Write your final response NOW based on those results.\n"
        )

    return f"""You are a helpful assistant with access to specialised **Skills**.

  ## Handling Requests

  1. Check if any skill matches the user's request using the descriptions below.
  2. If a skill matches, call `read_skill_instructions` ONCE, then call the skill tool ONCE.
  3. After tools return results, write your final Markdown response immediately.
  4. If no skill matches, answer from your own knowledge.

  ## ABSOLUTE Tool Usage Rules — violations cause infinite loops

  - Call `read_skill_instructions` EXACTLY ONCE per request — NEVER call it twice.
  - Call each skill tool EXACTLY ONCE — NEVER repeat a tool call.
  - If a tool returns an error, write the error to the user — do NOT retry with different args or a different skill.
  - After receiving ANY tool result (success OR error), STOP calling tools and write your response.
  - Do NOT fall back to a different skill if the first skill returned an error — report the error instead.
  - Do NOT call `web_page_scraper_tool` or any other skill as a workaround for a failed YouTube tool.

  ---

  {skills_block}
  {done_block}
  ---

  ## Response Format Rules

  - **ALWAYS** return clean Markdown text — never raw Python dicts, JSON objects, or repr strings.
  - **NEVER** include `extras`, `signature`, `type`, `id`, or base64 strings in your response.
  - When a tool returns an error, present it clearly: state the error, explain likely cause, suggest fix.
  - When a tool returns transcript text, format it clearly for the user.
  - Execute immediately — do not ask for confirmation.
  """

# ══════════════════════════════════════════════════════════════════════════════
# FAST PATH — handle "list skills" without going through the LLM round-trip
# This eliminates the risk of Gemini serialising the response object.
# ══════════════════════════════════════════════════════════════════════════════

_LIST_SKILLS_PATTERNS = [
    "what skills", "which skills", "list skills", "available skills",
    "what can you do", "what capabilities", "what tools", "show skills",
    "skills do you have", "skills available", "help me",
]

def _is_list_skills_query(query: str) -> bool:
    q = query.lower().strip()
    return any(p in q for p in _LIST_SKILLS_PATTERNS)

# Tracks (tool_name, args_json) pairs already executed in this run
# to prevent Gemini from calling the same tool twice.
_ALREADY_CALLED: set = set()

def tool_node(state: AgentState) -> AgentState:
    """Tool execution node — runs each tool call and returns ToolMessages."""
    last_msg           = state["messages"][-1]
    tool_results       = state.get("tool_results", [])
    skill_instructions = state.get("skill_instructions")
    new_messages       = []

    # Build a set of (name, args_json) already called this run from tool_results
    already_called = {
        (tr["tool"], json.dumps(tr["args"], sort_keys=True))
        for tr in tool_results
    }

    for tc in last_msg.tool_calls:
        name    = tc["name"]
        args    = tc["args"]
        call_id = tc["id"]
        call_key = (name, json.dumps(args, sort_keys=True))

        # ── Deduplication: skip if this exact call already ran ──────────────
        if call_key in already_called:
            print(f"[Tool] ⏭ SKIPPED duplicate call: {name}({json.dumps(args)[:80]})")
            # Return the cached result from the previous identical call
            cached = next(
                (tr["result_full"] for tr in tool_results
                 if tr["tool"] == name and
                 json.dumps(tr["args"], sort_keys=True) == json.dumps(args, sort_keys=True)),
                json.dumps({"note": f"Already called {name} — using previous result."})
            )
            new_messages.append(
                ToolMessage(content=cached, tool_call_id=call_id, name=name)
            )
            continue

        print(f"[Tool] → {name}({json.dumps(args)[:120]})")

        if name in TOOL_MAP:
            try:
                result = TOOL_MAP[name].invoke(args)
            except Exception as e:
                result = json.dumps({"error": str(e), "tool": name})
        else:
            result = json.dumps({"error": f"Unknown tool: {name}"})

        # Ensure result is always a plain string
        result = extract_text_content(result) if not isinstance(result, str) else result

        print(f"[Tool] ← {result[:200]}")

        if name == "read_skill_instructions":
            skill_instructions = result

        tool_results.append({
            "tool":           name,
            "args":           args,
            "result_preview": result[:500],   # for display
            "result_full":    result,          # full result for fallback rendering
        })
        already_called.add(call_key)

        new_messages.append(
            ToolMessage(content=cached if False else result, tool_call_id=call_id, name=name)
        )

    return {
        "messages":           new_messages,
        "selected_skill":     state.get("selected_skill"),
        "skill_instructions": skill_instructions,
        "tool_results":       tool_results,
        "final_response":     state.get("final_response"),
        "token_usage":        state.get("token_usage") or {},
    }


def agent_node(llm,state: AgentState, registry: Optional[Dict] = None) -> AgentState:
    """Main reasoning node — LLM decides what tool to call next (or ends)."""
    llm           =  llm 
    executed      = state.get("tool_results", [])   # tools already run this turn
    system_prompt = build_system_prompt(registry, executed_tools=executed if executed else None)
    messages      = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    selected_skill = state.get("selected_skill")
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "read_skill_instructions":
                selected_skill = tc["args"].get("skill_name", selected_skill)

    # Accumulate token usage across all agent turns
    this_usage = extract_token_usage(response)
    cumulative = merge_usage(state.get("token_usage") or {}, this_usage)

    return {
        "messages":           [response],
        "selected_skill":     selected_skill,
        "skill_instructions": state.get("skill_instructions"),
        "tool_results":       state.get("tool_results", []),
        "final_response":     state.get("final_response"),
        "token_usage":        cumulative,
    }


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent",agent_node)
    graph.add_node("execute_tools",tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue,
        {"execute_tools": "execute_tools", "end": END}
    )
    graph.add_edge("execute_tools", "agent")
    return graph.compile()


def run_agent(
    llm,
    user_query: str,
    verbose:bool = True,
    registry:   Optional[Dict] = None,
    ) -> Dict:
    
    if registry is None:
        registry = get_registry()

    if verbose:
        print(f"\n{'='*60}\nQUERY : {user_query}\nSKILLS: {list(registry.keys())}\n{'='*60}")
    
    # ── Fast path: listing skills never needs an LLM round-trip ──────────────
    response = is_skill_query_chain(llm,user_query)
    
    # if _is_list_skills_query(user_query):
    if response['result']: 
      response_text = list_available_skills.invoke({})
      if verbose:
        print(f"\nFAST PATH — list_available_skills\n{response_text}")
      return {
          "response":       response_text,
          "selected_skill": None,
          "tools_called":   ["list_available_skills"],
        }
    
    def agent_node_with_registry(state: AgentState) -> AgentState:
      return agent_node(llm,state,registry=registry)
  
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node_with_registry)
    graph.add_node("execute_tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue,
        {"execute_tools": "execute_tools", "end": END}
    )
    graph.add_edge("execute_tools", "agent")
    try:
      compiled = graph.compile()
    except Exception as e:
      print(f"[SkillAgent] Graph compilation failed: {e}")
        
    initial_state: AgentState = {
        "messages":           [HumanMessage(content=user_query)],
        "selected_skill":     None,
        "skill_instructions": None,
        "tool_results":       [],
        "final_response":     None,
        "token_usage":        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }

    final_state = compiled.invoke(initial_state, config={"recursion_limit": 12})

    # ── Extract clean text from the final message ─────────────────────────────
    last_msg      = final_state["messages"][-1]
    raw_content   = getattr(last_msg, "content", str(last_msg))
    response_text = extract_text_content(raw_content)
    
    tools_used = [t["tool"] for t in final_state.get("tool_results", [])]

    if verbose:
        print(f"\n{'='*60}\nRESPONSE:\n{'='*60}\n{response_text}")
        print(f"\nSkill : {final_state.get('selected_skill')}")
        print(f"Tools : {tools_used}")

    return {
        "response":       response_text,
        "selected_skill": final_state.get("selected_skill"),
        "tools_called":   tools_used,
        "tool_results":   final_state.get("tool_results", []),
        "token_usage":    final_state.get("token_usage") or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    from langchain_groq import ChatGroq
    os.environ["GROQ_API_KEY"] = 'YOUR_GROQ_KEY'
    llm_groq = ChatGroq(model="qwen/qwen3-32b")
    #result = run_agent(llm_groq, "What available skills do you have ?", verbose=True)
    result = run_agent(llm_groq, "who is president of Iran ?", verbose=True)
    print(result)
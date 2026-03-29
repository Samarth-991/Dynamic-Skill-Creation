# Dynamic Skill Generation — LangChain Skills Agent

A self-extending AI agent that writes, registers, and hot-reloads its own tools at runtime —
no restarts, no manual code edits. Describe a capability in plain English (or paste a docs URL,
or upload a markdown file) and the system generates a working LangChain skill, registers it,
and makes it available to chat immediately.

The repo supports multiple models and LLM service, including ollama support

---

## Table of Contents

1. [What is Dynamic Skill Generation?](#what-is-dynamic-skill-generation)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
   - [app.py — Streamlit UI](#apppy--streamlit-ui)
   - [skill_agent.py — LangGraph Agent](#skill_agentpy--langgraph-agent)
   - [skills_registry.py — Skill Discovery](#skills_registrypy--skill-discovery)
   - [create_skill.py — Skill Creator Pipeline](#create_skillpy--skill-creator-pipeline)
5. [The 8-Step Skill Creation Pipeline](#the-8-step-skill-creation-pipeline)
6. [Key Integration Points](#key-integration-points)
   - [Fresh Registry on Every Call](#fresh-registry-on-every-call)
   - [Hot-Reload After Creation](#hot-reload-after-creation)
   - [Shared SkillCreator Class](#shared-skillcreator-class)
7. [Skill Input Modes](#skill-input-modes)
8. [Skill Folder Structure](#skill-folder-structure)
9. [Data Flow](#data-flow)
10. [Installation](#installation)
11. [Usage](#usage)
    - [Streamlit UI](#streamlit-ui)
    - [CLI](#cli)
    - [Programmatic API](#programmatic-api)
12. [Environment Variables](#environment-variables)
13. [Adding Skills Manually](#adding-skills-manually)
14. [Token Usage Tracking](#token-usage-tracking)
15. [Design Decisions](#design-decisions)

---

## What is Dynamic Skill Generation?

Most LangChain agents are static — their tool list is defined at startup and never changes.
This project inverts that model. The agent can generate new tools for itself on demand:

1. A user describes a capability ("extract tables from PDFs")
2. The system uses an LLM to write a `SKILL.md` (the tool spec), a Python implementation
   script, and a `@tool` stub for the agent
3. Files are written to disk, the stub is injected into `skill_agent.py`, and
   `skill_agent.py` is hot-reloaded in memory
4. The new skill is live in the Chat tab — **no Streamlit restart, no manual code change**

This is the same pattern used by Claude Code's internal skill-creator system, re-implemented
Here with LangChain, LangGraph, and Streamlit.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          app.py  (Streamlit UI)                          │
│                                                                          │
│  Tab 1: 💬 Chat          Tab 2: 🛠️ Create Skill   Tab 3: 📦 Library    │
│  ─────────────────        ─────────────────────    ─────────────────     │
│  User sends query         User describes skill      Browse all skills    │
│        │                        │                   with SKILL.md        │
│        │                        │                   and script previews  │
│        ▼                        ▼                                        │
│  get_registry()          create_skill_programmatic()                     │
│  run_agent(query,              │                                         │
│    registry)                   ├── build_brief_from_description()        │
│        │                       ├── generate_skill_md()                   │
│        │                       ├── generate_script()                     │
│        │                       ├── generate_tool_stub()                  │
│        │                       ├── write_to_disk()                       │
│        │                       ├── register_tool()                       │
│        │                       ├── test_routing()                        │
│        │                       └── reload_tools()  ←── hot-reload        │
│        │                              │                                  │
│        │                    Skill immediately live                       │
│        │                    in Tab 1 Chat ─────────────────────►         │
└────────┼────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│              skill_agent.py                    │
│                                                │
│  run_agent(query, registry)                    │
│       │                                        │
│       ▼                                        │
│  LangGraph StateGraph                          │
│       │                                        │
│  agent_node ──► execute_tools ──► agent_node   │
│       │              │                 │       │
│  (routing)    (tool calls)       (response)    │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│           skills_registry.py                   │
│                                                │
│  get_registry()  ←── always reads fresh disk  │
│       │                                        │
│  format_skills_for_prompt()  → system prompt  │
│  get_skill_instructions()    → SKILL.md body  │
└────────────────────────────────────────────────┘
```

---

## Project Structure

```
project-root/
│
├── app.py                        # Streamlit UI — 3-tab interface
│
├── agent/
│   └── skill_agent.py            # LangGraph ReAct agent + TOOLS_LIST
│
├── skills/
│   ├── skills_registry.py        # Disk-based skill discovery
│   │
│   ├── browser-search/           # Example skill (one folder per skill)
│   │   ├── SKILL.md              # Spec: frontmatter + workflow + trigger phrases
│   │   ├── scripts/
│   │   │   └── browser_search.py # Implementation
│   │   └── requirements.txt
│   │
│   └── <skill-name>/             # Every new skill gets its own folder
│       ├── SKILL.md
│       ├── scripts/
│       │   └── <skill_name>.py
│       └── requirements.txt
│
├── create_skill/
│   └── create_skill.py           # SkillCreator class + CLI entry point
│
├── test_agent.py                 # Headless test runner
└── requirements.txt
```

---

## Core Components

### app.py — Streamlit UI

Three-tab interface that ties the whole system together.

| Tab | Purpose |
|-----|---------|
| **💬 Chat** | Conversational interface. Calls `get_registry()` + `run_agent()` on every message — always sees the latest skills. |
| **🛠️ Create Skill** | Describe a skill (text, URL, or markdown file upload). Runs the full 8-step pipeline, then calls `reload_tools()` so the new skill is immediately usable in Chat. |
| **📦 Library** | Browse every skill in the registry. Shows `SKILL.md` content, implementation script preview, trigger phrases, and requirements. |

After `create_skill_programmatic()` returns, `app.py` calls `reload_tools()`:

```python
# app.py (post-creation)
result = create_skill_programmatic(description, log=st.write)
reload_tools()   # re-imports skill_agent.py, rebuilds TOOLS + AGENT_GRAPH in memory
st.success(f"Skill '{result['skill_name']}' is now live in Chat.")
```

---

### skill_agent.py — LangGraph Agent

Hosts the `TOOLS_LIST` and the LangGraph `StateGraph` that powers the Chat tab.

```python
# Structure inside skill_agent.py
TOOLS_LIST = [
    existing_tool_1,
    existing_tool_2,
    # ← new @tool stubs are injected here by register_tool()
]

def run_agent(query: str, registry: dict = None) -> str:
    """
    Build and run the ReAct agent graph for a single query.
    Accepts an optional registry so routing always reflects the current skill set.
    """
    ...
```

The graph follows a standard ReAct loop:

```
agent_node  →  execute_tools  →  agent_node  →  (END when no tool calls remain)
```

`reload_tools()` re-imports this module and rebuilds `TOOLS`, `TOOL_MAP`, and `AGENT_GRAPH.`
in the Streamlit process memory, so newly injected stubs become callable without a restart.

---

### skills_registry.py — Skill Discovery

Reads the `skills/` directory from disk on every call — no caching, no stale state.

```python
# Key functions
get_registry() -> dict
    # Walks skills/, reads each SKILL.md frontmatter
    # Returns {skill_name: {description, trigger_phrases, script_path, ...}}

format_skills_for_prompt(registry: dict) -> str
    # Formats registry as a block injected into the agent system prompt
    # Tells the LLM which skills exist and when to use each one

get_skill_instructions(skill_name: str) -> str
    # Returns full SKILL.md body for a specific skill
    # Used by the agent to get step-by-step instructions before calling a tool
```

Because `get_registry()` always reads the disk, a skill created 1 second ago is already
visible to the very next `run_agent()` call.

---

### create_skill.py — Skill Creator Pipeline

Exposes two interfaces:

**`SkillCreator` class** — every pipeline step as an individual method, for power users:

```python
from create_skill import SkillCreator

creator = SkillCreator()
brief   = creator.build_brief_from_description("resize images to 512x512")
# or: brief = creator.build_brief_from_content(raw_markdown, source_hint="README.md")

skill_md    = creator.generate_skill_md(brief)
script_code = creator.generate_script(brief)
tool_stub   = creator.generate_tool_stub(brief)
skill_dir   = creator.write_to_disk(brief, skill_md, script_code)
registered  = creator.register_tool(tool_stub, brief["skill_name"])
passed, msg = creator.test_routing(brief)
```

**`create_skill_programmatic()` convenience function** — used by `app.py` and `test_agent.py`:

```python
from create_skill import create_skill_programmatic

result = create_skill_programmatic(
    description="resize images to 512x512",
    log=st.write,          # or print, or any callable
)
# result keys: skill_name, skill_dir, skill_md, script_code,
#              tool_stub, registered, test_passed, test_reason, token_usage
```

---

## The 8-Step Skill Creation Pipeline

Each step is a separate LLM call or I/O operation, making them individually testable and
replaceable.

```
Step 1:  Capture Intent
        └── LLM reads the description / URL content / markdown file
            and returns a structured JSON brief:
            { skill_name, one_liner, what_it_does, trigger_phrases,
              input_type, output_type, python_libraries, needs_script,
              suggested_test_query }

Step 2:  Generate SKILL.md
        └── LLM writes complete frontmatter + workflow doc from the brief
            Includes: Overview, Workflow steps, Usage Patterns,
            Error Handling table, Output Formatting rules, Best Practices

Step 3  Generate Implementation Script
        └── LLM writes a complete, working Python script
            Main function signature: run_<skill_name>(input_value: str) -> dict
            Includes CLI entry point and proper error handling

Step 4:  Generate @tool Stub
        └── LLM writes the LangChain @tool wrapper for skill_agent.py
            Handles sys.path manipulation, import, call, and cleanup

Step 5  Write to Disk
        └── Creates skills/<skill-name>/ folder tree:
            SKILL.md, scripts/<skill_name>.py, requirements.txt
            (backs up existing folder if name already exists)

Step 6  Register Tool
        └── Injects the @tool stub into skill_agent.py just before TOOLS_LIST
            Appends function name to TOOLS_LIST
            No-op if already registered (idempotent)

Step 7  Self-Test Routing
        └── Asks the LLM: "given the current skill registry, would you route
            '<suggested_test_query>' to '<skill_name>'?"
            Returns (passed: bool, reason: str)
            Warns if routing might be ambiguous

Step 8  Review & Iterate  (CLI only)
        └── Shows first 25 lines of SKILL.md and script
            Options: Accept / Redo SKILL.md / Redo Script / Redo Both /
                     Edit Description / Quit
```

---

## Key Integration Points

### Fresh Registry on Every Call

`run_agent()` accepts an optional `registry` parameter. Both `app.py` and `test_agent.py`
call `get_registry()` and pass it in on every invocation:

```python
# app.py — Chat tab
registry = get_registry()           # reads disk fresh every time
response = run_agent(query, registry=registry)
```

This means a skill created 1 second ago — whose files are now on disk — is already
visible to the very next agent call. There is no in-memory skills cache that could
go stale between creation and use.

---

### Hot-Reload After Creation

After `create_skill_programmatic()` writes files and injects the `@tool` stub into
`skill_agent.py`, `app.py` calls `reload_tools()`:

```python
# Inside reload_tools() — conceptually
import importlib
import skill_agent

importlib.reload(skill_agent)

# Rebuild module-level globals from the freshly imported module
TOOLS      = skill_agent.TOOLS_LIST
TOOL_MAP   = {t.name: t for t in TOOLS}
AGENT_GRAPH = skill_agent.build_graph(TOOLS)
```

This re-imports `skill_agent.py` (which now contains the new `@tool` stub), rebuilds
`TOOLS`, `TOOL_MAP`, and `AGENT_GRAPH` in the Streamlit process memory. The new skill
becomes callable in the Chat tab **with no Streamlit restart**.

```
create_skill_programmatic()
        │
        ├── writes  skills/<name>/SKILL.md
        ├── writes  skills/<name>/scripts/<name>.py
        ├── injects @tool stub → skill_agent.py
        │
        └── app.py calls reload_tools()
                    │
                    └── skill immediately callable in Chat tab ✓
```

---

### Shared SkillCreator Class

`create_skill.py` exposes both the full `SkillCreator` class (every pipeline step as
an individual method) and the `create_skill_programmatic()` convenience wrapper.

- **`app.py`** and **`test_agent.py`** use `create_skill_programmatic()` — single call,
  Returns a result dict with everything, including `token_usage`
- **Power users/custom pipelines** can import `SkillCreator` directly and call only the
  steps they need (e.g. regenerating just the script, or testing routing in isolation)

```python
# Convenience function — used by app.py
from create_skill import create_skill_programmatic
result = create_skill_programmatic(description, log=st.write)

# Direct class usage — for fine-grained control
from create_skill import SkillCreator
creator = SkillCreator()
brief   = creator.build_brief_from_description("parse YAML config files")
stub    = creator.generate_tool_stub(brief)   # just the stub, nothing else
```

---

## Skill Input Modes

The Create Skill tab supports three ways to define a new skill:

| Mode | How it works | Best for |
|------|-------------|----------|
| **✏️ Text Description** | Plain English description → `build_brief_from_description()` | Quick skill creation from scratch |
| **🌐 URL** | Fetches page via Firecrawl (or requests fallback) → `build_brief_from_content()` | Creating skills from existing docs / READMEs |
| **📄 Markdown File** | Uploads `.md` / `.txt` → `build_brief_from_content()` | Offline docs, internal wikis, local README files |

All three modes converge to the same `brief` dict, so `run_full_pipeline()` is
identical regardless of input source.

---

## Skill Folder Structure

Every skill lives in its own self-contained folder:

```
skills/
└── my-skill/
    ├── SKILL.md              ← Routing spec + full workflow instructions
    │     ---
    │     name: my-skill
    │     description: "Trigger phrases and when to use this skill..."
    │     ---
    │     # My Skill
    │     ## Overview ...
    │     ## Workflow ...
    │     ## Error Handling ...
    │
    ├── scripts/
    │   └── my_skill.py       ← Implementation
    │         def run_my_skill(input_value: str) -> dict:
    │             ...
    │         if __name__ == "__main__":
    │             ...
    │
    └── requirements.txt      ← Skill-specific pip dependencies
```

The `SKILL.md` frontmatter is parsed by `skills_registry.py` to build the routing
system prompt. The `description` field is the most important — it is keyword-rich
and tells the LLM exactly when to select this skill.

---

## Data Flow

### Chat tab — query to response

```
User query
    │
    ▼
get_registry()              ← reads skills/ from disk (always fresh)
    │
    ▼
format_skills_for_prompt()  ← builds system prompt block listing all skills
    │
    ▼
run_agent(query, registry)
    │
    ▼
LangGraph StateGraph
    │
    ├── agent_node          ← LLM decides which tool to call
    │       │
    ├── execute_tools       ← calls the @tool function
    │       │
    │       ├── sys.path.insert → import script → run_<skill>()
    │       │
    │       └── sys.path.remove (cleanup)
    │
    └── agent_node          ← LLM synthesises final answer
            │
            ▼
        Response to user
```

### Create Skill tab — description to live tool

```
User input (text / URL / file)
    │
    ▼
build_brief_from_description()  or  build_brief_from_content()
    │                                        ↑
    │                               fetch_url_content()
    │                               read_markdown_file()
    ▼
Brief dict (JSON)
    │
    ├── generate_skill_md()     → SKILL.md
    ├── generate_script()       → <skill_name>.py
    ├── generate_tool_stub()    → @tool function string
    ├── write_to_disk()         → skills/<name>/ folder tree
    ├── register_tool()         → injects stub into skill_agent.py
    ├── test_routing()          → verifies LLM would route correctly
    └── reload_tools()          → hot-reloads skill_agent.py in memory
                │
                ▼
        Skill live in Chat tab ✓
```

---

## Installation

```bash
git clone https://github.com/your-org/dynamic-skill-agent
cd dynamic-skill-agent

pip install -r requirements.txt
```

Core dependencies:

```
langchain-core
langchain-community
langchain-groq          # LLM backend (swap for langchain-anthropic, langchain-openai, etc.)
langgraph
streamlit
firecrawl-py            # URL fetching (optional — falls back to requests+bs4)
beautifulsoup4
requests
```

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Three tabs are available immediately:

- **Chat** — talk to the agent; it uses whatever skills exist on disk right now
- **Create Skill** — describe a new skill; it will be live in Chat within seconds
- **Library** — browse all skills, read their specs and implementation scripts

### CLI

Interactive interview mode (prompts for each field, then lets you review and iterate):

```bash
python create_skill/create_skill.py
```

Single-shot with a description:

```bash
python create_skill/create_skill.py --skill "extract all URLs from a webpage"
```

Skip routing self-test:

```bash
python create_skill/create_skill.py --skill "convert CSV to JSON" --no-test
```

### Programmatic API

```python
from create_skill.create_skill import create_skill_programmatic, SkillCreator
from skills.skills_registry import get_registry
from agent.skill_agent import run_agent, reload_tools

# Create a skill
result = create_skill_programmatic(
    description="translate text between languages using deep-translator",
    log=print,
)
print(result["skill_name"])      # e.g. "text-translator"
print(result["test_passed"])     # True / False
print(result["token_usage"])     # {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}

# Hot-reload so the new skill is callable
reload_tools()

# Use it immediately
registry = get_registry()
response = run_agent("Translate 'hello world' to French", registry=registry)
print(response)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | LLM backend for all generation steps |
| `FIRECRAWL_API_KEY` | No | URL fetching in Create Skill tab (falls back to requests if not set) |
| `SERPAPI_API_KEY` | No | Web search skill (only needed if browser-search skill is used) |

```bash
export GROQ_API_KEY="gsk_..."
export FIRECRAWL_API_KEY="fc-..."
export SERPAPI_API_KEY="..."
```

---

## Adding Skills Manually

You can also create a skill by hand without using the generator:

1. Create `skills/<your-skill>/SKILL.md` with the required frontmatter:

```markdown
---
name: your-skill
description: "Trigger this skill when the user wants to... Include keywords: X, Y, Z."
---

# Your Skill

## Overview
...

## Workflow
Step 1: ...
Step 2: ...
```

2. Create `skills/<your-skill>/scripts/your_skill.py` with:

```python
def run_your_skill(input_value: str) -> dict:
    # your implementation
    return {"success": True, "result": ...}
```

3. Add a `@tool` stub to `skill_agent.py` before `TOOLS_LIST`:

```python
@tool
def your_skill_tool(input_value: str) -> str:
    """One-sentence description used by the LLM for routing."""
    scripts_dir = Path(__file__).parent / "skills" / "your-skill" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import your_skill
        result = your_skill.run_your_skill(input_value)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "error_type": type(e).__name__})
    finally:
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))
```

4. Add `your_skill_tool` to `TOOLS_LIST`.

The skill will be picked up by `get_registry()` on the next call with no restart needed.

---

## Token Usage Tracking

Every call to `create_skill_programmatic()` tracks cumulative token usage across all 8
LLM calls in the pipeline. The result dict always includes:

```python
result["token_usage"] = {
    "input_tokens":  1234,
    "output_tokens": 567,
    "total_tokens":  1801,
}
```

This is accumulated by `_accumulate_tokens()` across Steps 1–7 and reset at the
start of each new `create_skill_programmatic()` call via `_reset_token_counter()`.

---

## Design Decisions

**Why read the registry from disk on every call?**
Skills can be created at any time — by the UI, by the CLI, or by `test_agent.py` running
in a separate process. An in-memory cache would require invalidation logic and would break
in multi-process scenarios. Reading disk is fast enough at this scale and eliminates
an entire class of stale-state bugs.

**Why inject the @tool stub as a string into skill_agent.py?**
LangChain's `@tool` decorator needs the function to exist in the module's namespace at
import time. Writing the stub as source code and reloading the module is the cleanest
way to achieve this without metaclass tricks or runtime `exec()` in the agent itself.

**Why use a per-skill `scripts/` subfolder?**
Each skill's `sys.path.insert` + `import` + `sys.path.remove` pattern keeps skill
namespaces isolated. Two skills can both have a `utils.py` without collision, because
the path is inserted and removed around each tool call.

**Why LangGraph instead of a simple LangChain `AgentExecutor`?**
LangGraph gives explicit control over the agent loop as a `StateGraph`, making it
straightforward to add nodes (e.g. a skill-instruction-lookup node that reads the
relevant `SKILL.md` before a tool call) without restructuring the entire agent.

**Why Groq / Llama 4?**
Low latency on the skill creation steps matters — generating a brief + SKILL.md +
script + tool stub + routing test is 5 sequential LLM calls. Groq's inference speed
keeps the end-to-end creation time under ~15 seconds. The LLM backend is swappable
by changing `_llm_call()`.

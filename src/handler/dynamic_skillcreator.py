"""
create_skill.py

Mimics Claude Code's skill-creator pipeline. Can be used three ways:

  1. CLI (interactive):
        python create_skill.py
        python create_skill.py --skill "summarise PDF documents"

  2. Programmatic (called by app.py / test_agent.py):
        from create_skill import SkillCreator
        creator  = SkillCreator()
        brief    = creator.build_brief_from_description("extract text from PDFs")
        result   = creator.run_full_pipeline(brief, interactive=False)
        # result = {"skill_name", "skill_dir", "skill_md", "script_code",
        #           "tool_stub",  "registered", "test_passed"}

  3. Single-shot helper (used by app.py non-interactively):
        from create_skill import create_skill_programmatic
        result = create_skill_programmatic("describe the skill here")

The pipeline steps mirror Claude Code's skill-creator SKILL.md exactly:
  Step 1  Capture Intent       — LLM extracts a structured brief from the description
  Step 2  Generate SKILL.md    — LLM writes frontmatter + full workflow body
  Step 3  Generate script      — LLM writes a working Python implementation
  Step 4  Generate @tool stub  — LLM writes the LangChain @tool wrapper
  Step 5  Write to disk        — creates skills/<name>/ folder tree
  Step 6  Register tool        — injects stub into skill_agent.py
  Step 7  Self-test routing    — verifies the new skill would be triggered correctly
  Step 8  Review & iterate     — interactive loop (CLI only)
"""

import os
import re
import sys
import json
import shutil
import argparse
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).parent
SKILLS_DIR   = PROJECT_ROOT / "../skills"
AGENT_FILE   = PROJECT_ROOT / "agent/skill_agent.py"

# ── Terminal colours (CLI only) ───────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{RESET}"

# ── Global token accumulator for create_skill pipeline ──────────────────────
# Reset at the start of each create_skill_programmatic() call.
_CREATE_TOKENS: Dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def _reset_token_counter():
    global _CREATE_TOKENS
    _CREATE_TOKENS = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def get_create_token_usage() -> Dict:
    """Return a copy of the accumulated token usage for the last skill creation."""
    return dict(_CREATE_TOKENS)

def _extract_text(content) -> str:
    """
    Safely extract plain text from any Gemini/LangChain content format.
    Gemini 3 Pro Preview returns content as a list of dicts, not a plain string.
    Handles: str, list of {type, text} dicts, objects with .text attribute.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    if hasattr(content, "text"):
        return str(content.text).strip()
    return str(content).strip()

def _accumulate_tokens(response) -> Dict:
    """Extract token counts from a response and add them to the global counter."""
    global _CREATE_TOKENS
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    meta  = getattr(response, "response_metadata", {}) or {}
    um    = meta.get("usage_metadata") or meta.get("token_counts") or {}
    if um:
        usage["input_tokens"]  = um.get("prompt_token_count")     or um.get("input_tokens",  0)
        usage["output_tokens"] = um.get("candidates_token_count") or um.get("output_tokens", 0)
        usage["total_tokens"]  = um.get("total_token_count")       or um.get("total_tokens",  0)
    if usage["total_tokens"] == 0 and hasattr(response, "usage_metadata"):
        umd = response.usage_metadata or {}
        usage["input_tokens"]  = umd.get("input_tokens",  0)
        usage["output_tokens"] = umd.get("output_tokens", 0)
        usage["total_tokens"]  = umd.get("total_tokens",  0)
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    _CREATE_TOKENS["input_tokens"]  += usage["input_tokens"]
    _CREATE_TOKENS["output_tokens"] += usage["output_tokens"]
    _CREATE_TOKENS["total_tokens"]  += usage["total_tokens"]
    return usage

def _llm_call(system: str, user: str, temperature: float = 0.2) -> str:
    from langchain_groq import ChatGroq
    os.environ["GROQ_API_KEY"] = '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'
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
    llm = ChatGroq(model=model_name)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    _accumulate_tokens(response)
    return _extract_text(response.content)


def _strip_fences(text: str, lang: str = "") -> str:
    """Remove markdown code fences from LLM output."""
    text = re.sub(rf"^```{lang}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*$",       "", text, flags=re.MULTILINE)
    return text.strip()


class SkillCreator:
    """
    Encapsulates every step of the skill-creation pipeline.
    Can be used interactively (CLI) or programmatically (app.py / test_agent.py).
    """
    def __init__(self,llm=None):
        self._llm_call = llm
    
    def build_brief_from_description(self, description: str) -> Dict:
        """
        Use LLM to extract a structured brief from a free-text description.
        Returns a dict with all fields needed for generation.
        Does NOT prompt the user — safe to call from a UI.
        """
        raw = _llm_call(
            system=textwrap.dedent("""
                You are a skill architect. Given a skill description, return ONLY valid JSON
                (no markdown fences, no explanation) matching this schema exactly:
                {
                  "skill_name":          "lowercase-hyphen-name",
                  "one_liner":           "one sentence summary",
                  "what_it_does":        "detailed description",
                  "trigger_phrases":     ["phrase 1", "phrase 2"],
                  "input_type":          "what the user provides",
                  "output_type":         "what the skill produces",
                  "python_libraries":    ["lib1", "lib2"],
                  "needs_script":        true,
                  "suggested_test_query":"a realistic user query that should trigger this skill"
                }
            """),
            user=f"Skill description: {description}",
            temperature=0.1,
        )
        raw = _strip_fences(raw, "json")
        try:
            brief = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: derive a minimal brief from the description text
            slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")[:40]
            brief = {
                "skill_name":          slug,
                "one_liner":           description,
                "what_it_does":        description,
                "trigger_phrases":     [description],
                "input_type":          "text",
                "output_type":         "text",
                "python_libraries":    [],
                "needs_script":        True,
                "suggested_test_query": description,
            }

        # Normalise
        brief["skill_name"] = re.sub(r"[^a-z0-9-]", "-",
                                      brief["skill_name"].lower()).strip("-")
        if isinstance(brief.get("trigger_phrases"), str):
            brief["trigger_phrases"] = [brief["trigger_phrases"]]
        if isinstance(brief.get("python_libraries"), str):
            brief["python_libraries"] = [l.strip() for l in
                                          brief["python_libraries"].split(",") if l.strip()]
        if isinstance(brief.get("needs_script"), str):
            brief["needs_script"] = brief["needs_script"].lower() in ("yes","true","1","y")

        return brief

    
    def build_brief_from_content(self, content: str, source_hint: str = "") -> Dict:
        """
        Build a structured brief from raw content (fetched URL or uploaded markdown).
        The LLM reads the full content and infers skill intent from it.

        Args:
            content     : Raw text/markdown extracted from URL or file
            source_hint : Optional label shown to LLM e.g. "GitHub README", "docs page"
        """
        # Truncate to avoid token overflow — 6000 chars is safe for most models
        truncated = content[:6000]
        if len(content) > 6000:
            truncated += "\n\n[... content truncated for brevity ...]"

        raw = _llm_call(
            system=textwrap.dedent("""
                You are a skill architect. You will receive content from a documentation page,
                README, or markdown file. Your job is to infer what skill this describes and
                return ONLY valid JSON (no fences, no explanation) matching this schema exactly:
                {
                "skill_name":          "lowercase-hyphen-name",
                "one_liner":           "one sentence summary",
                "what_it_does":        "detailed description of what the skill does",
                "trigger_phrases":     ["phrase 1", "phrase 2", "phrase 3"],
                "input_type":          "what the user provides",
                "output_type":         "what the skill produces",
                "python_libraries":    ["lib1", "lib2"],
                "needs_script":        true,
                "suggested_test_query":"a realistic user query that should trigger this skill"
                }
                Infer all fields from the content. Be specific with trigger_phrases.
            """),
            user=textwrap.dedent(f"""
                Source: {source_hint or 'uploaded content'}

                Content:
                {truncated}
            """),
            temperature=0.1,
        )
        raw = _strip_fences(raw, "json")
        try:
            brief = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback to description-based brief using the first 200 chars as hint
            return self.build_brief_from_description(content[:200])

        # Normalise (same as build_brief_from_description)
        brief["skill_name"] = re.sub(r"[^a-z0-9-]", "-",
                                    brief["skill_name"].lower()).strip("-")
        if isinstance(brief.get("trigger_phrases"), str):
            brief["trigger_phrases"] = [brief["trigger_phrases"]]
        if isinstance(brief.get("python_libraries"), str):
            brief["python_libraries"] = [l.strip() for l in
                                        brief["python_libraries"].split(",") if l.strip()]
        if isinstance(brief.get("needs_script"), str):
            brief["needs_script"] = brief["needs_script"].lower() in ("yes","true","1","y")

        return brief
        

    
    def interview_user(self, description: str) -> Dict:
        """
        Interactive version of build_brief_from_description.
        Prints the inferred brief and lets the user correct every field.
        CLI-only — do not call from a UI.
        """
        brief = self.build_brief_from_description(description)

        print(_c(CYAN, "\nHere's what I understood about your skill:\n"))

        fields = [
            ("skill_name",           "Skill name (folder name)",            "my-skill"),
            ("one_liner",            "One-line summary",                    "Does something useful"),
            ("what_it_does",         "Full description",                    "Detailed explanation"),
            ("trigger_phrases",      "Trigger phrases (comma-separated)",   "help with X, do Y"),
            ("input_type",           "Input type",                          "text / URL / file"),
            ("output_type",          "Output type",                         "text / file / JSON"),
            ("python_libraries",     "Python libraries (comma-separated)",  "requests"),
            ("needs_script",         "Needs a Python script? (yes/no)",     "yes"),
            ("suggested_test_query", "Sample test query",                   "Do X with this: ..."),
        ]

        gathered = {}
        for key, label, default in fields:
            inferred = brief.get(key, "")
            if isinstance(inferred, list):
                inferred_display = ", ".join(inferred)
            elif isinstance(inferred, bool):
                inferred_display = "yes" if inferred else "no"
            else:
                inferred_display = str(inferred) if inferred else ""

            prompt = f"  {_c(CYAN, label)}"
            if inferred_display:
                prompt += f" [{_c(GREEN, inferred_display)}]"
            prompt += ": "

            val = input(prompt).strip()
            if not val:
                val = inferred_display or default
            gathered[key] = val

        # Re-normalise after user edits
        gathered["skill_name"] = re.sub(r"[^a-z0-9-]", "-",
                                         gathered["skill_name"].lower()).strip("-")
        if isinstance(gathered["trigger_phrases"], str):
            gathered["trigger_phrases"] = [t.strip() for t in
                                            gathered["trigger_phrases"].split(",")]
        if isinstance(gathered["python_libraries"], str):
            gathered["python_libraries"] = [l.strip() for l in
                                             gathered["python_libraries"].split(",") if l.strip()]
        gathered["needs_script"] = gathered.get("needs_script", "yes").lower() in (
            "yes","true","1","y"
        )
        return gathered

    # Generate SKILL.md

    _SKILL_MD_SYSTEM = textwrap.dedent("""
        You are an expert skill architect for an AI agent system.
        Write a complete, production-quality SKILL.md for a new skill.

        FORMAT:
        ---
        name: <skill-name>
        description: <1-2 sentence description. State WHEN to trigger it and WHAT it does.
                     Include specific keywords and contexts. Be slightly pushy — mention all
                     scenarios where this skill should be used.>
        ---

        # <Title>

        ## Overview
        ## Automatic Processing
        ## Core Capabilities
        ## Workflow  (Step 1, Step 2, ... — be very explicit, LLM follows exactly)
        ## Usage Patterns  (Pattern 1: Basic, Pattern 2: Advanced, ...)
        ## Error Handling  (table: Error | Cause | User Message)
        ## Output Formatting  (rules for short / medium / long content)
        ## Best Practices

        RULES:
        - Keep under 400 lines
        - Workflow steps must be explicit — no vague instructions
        - description field must be keyword-rich for routing
        - Return ONLY raw SKILL.md — no fences, no explanation
    """)

    def generate_skill_md(self, brief: Dict) -> str:
        libs = ", ".join(brief.get("python_libraries", [])) or "none"
        prompt = textwrap.dedent(f"""
            Create a complete SKILL.md for:
            Skill name:       {brief['skill_name']}
            Summary:          {brief['one_liner']}
            Full description: {brief['what_it_does']}
            Trigger phrases:  {', '.join(brief.get('trigger_phrases', []))}
            Input:            {brief['input_type']}
            Output:           {brief['output_type']}
            Libraries:        {libs}
            Has script:       {brief['needs_script']}
            Test query:       {brief['suggested_test_query']}
        """)
        return _llm_call(self._SKILL_MD_SYSTEM, prompt, temperature=0.15)


    _SCRIPT_SYSTEM = textwrap.dedent("""
        You are an expert Python developer writing an implementation script for an AI skill.
        Requirements:
        - Complete, working Python — no placeholders like "# your logic here"
        - Main function accepts primary input, returns dict with {"success": bool, ...}
        - Proper try/except error handling throughout
        - CLI entry point: if __name__ == "__main__"
        - Clear docstrings on every function
        - Use only the specified libraries (plus stdlib)
        Return ONLY raw Python code — no fences, no explanation.
    """)

    def generate_script(self, brief: Dict) -> str:
        fn = brief['skill_name'].replace('-', '_')
        libs = ", ".join(brief.get("python_libraries", [])) or "stdlib only"
        prompt = textwrap.dedent(f"""
            Write a complete Python implementation for:
            Skill:          {brief['skill_name']}
            What it does:   {brief['what_it_does']}
            Input:          {brief['input_type']}
            Output:         {brief['output_type']}
            Libraries:      {libs}
            Main function:  run_{fn}(input_value: str) -> dict
            Script file:    {fn}.py
        """)
        code = _llm_call(self._SCRIPT_SYSTEM, prompt, temperature=0.1)
        return _strip_fences(code, "python")


    _TOOL_STUB_SYSTEM = textwrap.dedent("""
        Write a LangChain @tool function for skill_agent.py.
        Rules:
        - Import the script from skills/<name>/scripts/ using sys.path manipulation
        - Call the main function and return json.dumps(result)
        - Handle ImportError and all exceptions with try/except/finally
        - One-sentence docstring (LLM reads this for routing decisions)
        - Correct argument type matching the skill input
        - Start with @tool — no imports at top level, no fences, no explanation
    """)

    def generate_tool_stub(self, brief: Dict) -> str:
        fn         = brief['skill_name'].replace('-', '_')
        script_file = f"{fn}.py"
        prompt = textwrap.dedent(f"""
            Write a @tool function for skill_agent.py:
            Skill name:    {brief['skill_name']}
            Function name: {fn}_tool
            Script:        skills/{brief['skill_name']}/scripts/{script_file}
            Main fn:       run_{fn}
            Input:         {brief['input_type']}
            One-liner:     {brief['one_liner']}

            Use this exact pattern:
            @tool
            def {fn}_tool(input_value: str) -> str:
                \"\"\"<one-sentence docstring>\"\"\"
                scripts_dir = Path(__file__).parent / "skills" / "{brief['skill_name']}" / "scripts"
                sys.path.insert(0, str(scripts_dir))
                try:
                    import {fn}
                    result = {fn}.run_{fn}(input_value)
                    return json.dumps(result, ensure_ascii=False, indent=2)
                except Exception as e:
                    return json.dumps({{"error": str(e), "error_type": type(e).__name__}})
                finally:
                    if str(scripts_dir) in sys.path:
                        sys.path.remove(str(scripts_dir))
        """)
        stub = _llm_call(self._TOOL_STUB_SYSTEM, prompt, temperature=0.1)
        return _strip_fences(stub, "python")

    def write_to_disk(self, brief: Dict, skill_md: str, script_code: str) -> Path:
        """Create skills/<name>/ folder with SKILL.md, scripts/, requirements.txt."""
        skill_dir   = SKILLS_DIR / brief["skill_name"]
        scripts_dir = skill_dir / "scripts"

        if skill_dir.exists():
            backup = skill_dir.parent / f"{brief['skill_name']}_backup"
            shutil.copytree(skill_dir, backup, dirs_exist_ok=True)
            shutil.rmtree(skill_dir)

        skill_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

        (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        script_name = f"{brief['skill_name'].replace('-', '_')}.py"
        (scripts_dir / script_name).write_text(script_code, encoding="utf-8")

        if brief.get("python_libraries"):
            (skill_dir / "requirements.txt").write_text(
                "\n".join(brief["python_libraries"]) + "\n", encoding="utf-8"
            )

        return skill_dir


    def register_tool(self, tool_stub: str, skill_name: str) -> bool:
        """
        Inject the @tool stub into skill_agent.py just before the TOOLS_LIST
        and add the function name to that list.
        Returns True if successfully registered.
        """
        if not AGENT_FILE.exists():
            return False

        content = AGENT_FILE.read_text(encoding="utf-8")
        fn_name = f"{skill_name.replace('-', '_')}_tool"

        if fn_name in content:
            return False  # already registered

        # Find TOOLS_LIST = [ and insert stub just before it
        match = re.search(r"^TOOLS_LIST\s*=\s*\[", content, re.MULTILINE)
        if not match:
            return False

        new_content = (
            content[:match.start()]
            + "\n\n" + tool_stub + "\n\n"
            + content[match.start():]
        )
        # Add to TOOLS_LIST
        new_content = new_content.replace(
            "TOOLS_LIST = [",
            f"TOOLS_LIST = [\n    {fn_name},"
        )
        AGENT_FILE.write_text(new_content, encoding="utf-8")
        return True

    def test_routing(self, brief: Dict) -> Tuple[bool, str]:
        """
        Ask Gemini whether it would route the suggested_test_query to this skill.
        Returns (passed: bool, reason: str).
        """
        from handler.skills_registry import get_registry, format_skills_for_prompt

        registry     = get_registry()
        skills_block = format_skills_for_prompt(registry)
        skill_name   = brief["skill_name"]

        if skill_name not in registry:
            return False, f"Skill '{skill_name}' not in registry after creation."

        raw = _llm_call(
            system=textwrap.dedent(f"""
                You are a routing system. Given a user query and available skills,
                return ONLY JSON:
                {{"selected_skill": "<skill-name or null>",
                  "confidence": "<high/medium/low>",
                  "reason": "<one sentence>"}}
                Available skills:{skills_block}
            """),
            user=f"User query: {brief['suggested_test_query']}",
            temperature=0.0,
        )
        raw = _strip_fences(raw, "json")
        try:
            r = json.loads(raw)
            selected   = r.get("selected_skill", "")
            confidence = r.get("confidence", "")
            reason     = r.get("reason", "")
            passed     = selected == skill_name
            msg        = f"Routed to '{selected}' ({confidence}). {reason}"
            return passed, msg
        except Exception as e:
            return False, f"Could not parse routing response: {e}"


    def interactive_review(
        self, brief: Dict, skill_dir: Path, skill_md: str, script_code: str
    ) -> Tuple[str, str]:
        """
        Show previews and let the user request regeneration or manual edits.
        CLI-only. Returns final (skill_md, script_code).
        """
        print(_c(BOLD, "\n── Review & Iterate ─────────────────────────────────────────\n"))
        while True:
            print(_c(CYAN, "Generated SKILL.md (first 25 lines):"))
            for i, line in enumerate(skill_md.splitlines()[:25], 1):
                print(f"  {i:3}  {line}")

            print(_c(CYAN, "\nGenerated script (first 25 lines):"))
            for i, line in enumerate(script_code.splitlines()[:25], 1):
                print(f"  {i:3}  {line}")

            print(_c(CYAN, "\n[enter] Accept  [1] Redo SKILL.md  [2] Redo script  "
                           "[3] Redo both  [4] Edit description  [q] Quit"))
            choice = input(_c(BOLD, "Choice: ")).strip().lower()

            if choice in ("", "enter"):
                break
            elif choice == "1":
                fb = input(_c(CYAN, "Feedback for SKILL.md: ")).strip()
                skill_md = _llm_call(
                    self._SKILL_MD_SYSTEM,
                    f"Rewrite based on feedback.\nOriginal:\n{skill_md}\nFeedback: {fb}",
                )
                (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
                print(_c(GREEN, "✔ SKILL.md regenerated."))
            elif choice == "2":
                fb = input(_c(CYAN, "Feedback for script: ")).strip()
                script_code = _strip_fences(
                    _llm_call(self._SCRIPT_SYSTEM,
                               f"Rewrite based on feedback.\nOriginal:\n{script_code}\nFeedback: {fb}",
                               temperature=0.1),
                    "python",
                )
                sname = f"{brief['skill_name'].replace('-','_')}.py"
                (skill_dir / "scripts" / sname).write_text(script_code, encoding="utf-8")
                print(_c(GREEN, "✔ Script regenerated."))
            elif choice == "3":
                fb = input(_c(CYAN, "Feedback for both: ")).strip()
                skill_md = _llm_call(
                    self._SKILL_MD_SYSTEM,
                    f"Rewrite based on feedback.\nOriginal:\n{skill_md}\nFeedback: {fb}",
                )
                script_code = _strip_fences(
                    _llm_call(self._SCRIPT_SYSTEM,
                               f"Rewrite based on feedback.\nOriginal:\n{script_code}\nFeedback: {fb}",
                               temperature=0.1),
                    "python",
                )
                (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
                sname = f"{brief['skill_name'].replace('-','_')}.py"
                (skill_dir / "scripts" / sname).write_text(script_code, encoding="utf-8")
                print(_c(GREEN, "✔ Both regenerated."))
            elif choice == "4":
                m = re.search(r"^description:\s*(.+)", skill_md, re.MULTILINE)
                print(_c(CYAN, f"Current: {m.group(1) if m else '(not found)'}"))
                new_desc = input(_c(CYAN, "New description: ")).strip()
                if new_desc and m:
                    skill_md = skill_md.replace(m.group(0), f"description: {new_desc}")
                    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
                    print(_c(GREEN, "✔ Description updated."))
            elif choice == "q":
                break
            else:
                print(_c(RED, "Invalid — try again."))

        return skill_md, script_code

    def run_full_pipeline(
        self,
        brief:       Dict,
        interactive: bool = False,
        log:         callable = print,
    ) -> Dict:
        """
        Run the complete skill creation pipeline.

        Args:
            brief       : Structured brief dict (from build_brief_from_description
                          or interview_user).
            interactive : If True, show review/iterate prompts (CLI mode).
                          If False, run silently — suitable for Streamlit.
            log         : Callable for progress messages (default: print).
                          Pass a Streamlit st.write or st.status.write for UI logging.

        Returns dict with keys:
            skill_name, skill_dir, skill_md, script_code,
            tool_stub, registered, test_passed, test_reason
        """
        skill_name = brief["skill_name"]

        log(f"⚙️  Generating SKILL.md for **{skill_name}**...")
        skill_md = self.generate_skill_md(brief)

        log("⚙️  Generating implementation script...")
        script_code = self.generate_script(brief) if brief.get("needs_script") else (
            "# No implementation script needed.\n"
        )

        log("⚙️  Generating @tool stub...")
        script_file = f"{skill_name.replace('-', '_')}.py"
        tool_stub   = self.generate_tool_stub(brief)

        log("💾  Writing skill to disk...")
        skill_dir = self.write_to_disk(brief, skill_md, script_code)
        log(f"✅  Skill folder created: `{skill_dir.relative_to(PROJECT_ROOT)}`")

        log("🔧  Registering @tool in skill_agent.py...")
        registered = self.register_tool(tool_stub, skill_name)
        log(f"{'✅' if registered else '⚠️ Already registered — '}  @tool {skill_name.replace('-','_')}_tool")

        if interactive:
            skill_md, script_code = self.interactive_review(
                brief, skill_dir, skill_md, script_code
            )

        log("🧪  Running routing self-test...")
        test_passed, test_reason = self.test_routing(brief)
        if test_passed:
            log(f"✅  Routing test passed — {test_reason}")
        else:
            log(f"⚠️  Routing test: {test_reason}")
            log("    Tip: Update the SKILL.md description with stronger trigger keywords.")

        return {
            "skill_name":   skill_name,
            "skill_dir":    skill_dir,
            "skill_md":     skill_md,
            "script_code":  script_code,
            "tool_stub":    tool_stub,
            "registered":   registered,
            "test_passed":  test_passed,
            "test_reason":  test_reason,
        }


def create_skill_programmatic(
    llm,
    description: str,
    log: callable = print,
) -> Dict:
    """
    Non-interactive entry point for app.py / test_agent.py.
    Builds brief from description, runs full pipeline, returns result dict.
    Also returns token_usage dict with total tokens consumed across all LLM calls.
    """
    _reset_token_counter()          # start fresh for this creation run
    creator = SkillCreator(llm=llm)
    log(f"📋  Building brief from description: \"{description}\"")
    brief = creator.build_brief_from_description(description)
    log(f"📋  Skill name resolved to: **{brief['skill_name']}**")
    result = creator.run_full_pipeline(brief, interactive=False, log=log)
    result["token_usage"] = get_create_token_usage()   # attach totals to result
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create a new skill for the LangChain Skills Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python create_skill.py
          python create_skill.py --skill "extract text from PDF files"
          python create_skill.py --skill "scrape a webpage and summarise it" --no-test
        """),
    )
    parser.add_argument("--skill", "-s", default="",
                        help="Description of the skill to create")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip routing self-test")
    args = parser.parse_args()

    description = args.skill.strip()
    if not description:
        print(_c(BOLD + CYAN, "\n🧠 LangChain Skill Creator\n"))
        description = input(_c(BOLD, "What skill do you want to create? ")).strip()
        if not description:
            print(_c(RED, "No description provided. Exiting."))
            sys.exit(1)

    print(_c(BOLD + CYAN, "\n╔══════════════════════════════════════════════╗"))
    print(_c(BOLD + CYAN,   "║     LangChain Skill Creator                  ║"))
    print(_c(BOLD + CYAN,   "╚══════════════════════════════════════════════╝\n"))

    creator = SkillCreator()

    # Interactive interview
    brief = creator.interview_user(description)
    print(_c(GREEN, f"\n✔  Skill name: {brief['skill_name']}\n"))

    # Full pipeline with interactive review
    result = creator.run_full_pipeline(brief, interactive=True)

    # Summary
    print(_c(BOLD + GREEN, "\n╔══════════════════════════════════════════════╗"))
    print(_c(BOLD + GREEN,   "║  ✔  Skill created successfully!              ║"))
    print(_c(BOLD + GREEN,   "╚══════════════════════════════════════════════╝\n"))
    print(_c(BOLD, "Files:"))
    for f in sorted(result["skill_dir"].rglob("*")):
        if f.is_file():
            print(f"  📄 {f.relative_to(PROJECT_ROOT)}")

    if brief.get("python_libraries"):
        print(_c(BOLD, "\nInstall dependencies:"))
        print(_c(CYAN, f"  pip install {' '.join(brief['python_libraries'])}"))

    print(_c(BOLD, "\nTest it:"))
    print(_c(CYAN, f"  python test_agent.py \"{brief['suggested_test_query']}\""))
    print(_c(BOLD, "\nLaunch UI:"))
    print(_c(CYAN, "  streamlit run app.py\n"))


if __name__ == "__main__":
    main()

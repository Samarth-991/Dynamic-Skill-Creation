"""
skills_registry.py

Scans the skills/ directory, parses SKILL.md frontmatter (name + description),
and builds a registry that the LangGraph agent uses for skill discovery and selection.

This mirrors exactly what Claude does:
  1. At startup, Claude reads all available SKILL.md descriptions
  2. Claude stores name, description, and file path for each skill
  3. When a user query arrives, Claude's routing logic matches the query to skill descriptions
  4. Claude then reads the FULL SKILL.md content and follows its workflow

The registry is intentionally NOT a module-level singleton anymore — it is always
loaded fresh via get_registry() so that newly created skills are immediately visible
without restarting the process.
"""

import re
from pathlib import Path
from typing import Dict, Optional

SKILLS_DIR = Path(__file__).parent / "../skills"


def parse_frontmatter(content: str) -> Dict[str, str]:
    """
    Parse YAML-style frontmatter from SKILL.md files.

    Claude skill files use a --- delimited frontmatter block containing:
      name: skill-name
      description: what this skill does (used for routing)

    Everything after the second --- is the full skill body (instructions/workflow).
    """
    frontmatter = {}
    body = content

    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx != -1:
            fm_block = content[3:end_idx].strip()
            body = content[end_idx + 3:].strip()
            for line in fm_block.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    frontmatter[key.strip()] = value.strip()

    frontmatter["_body"] = body
    return frontmatter


def load_skill_registry(silent: bool = False) -> Dict[str, Dict]:
    """
    Scan the skills/ directory and build a registry of all available skills.

    Always reads from disk — call this whenever you need an up-to-date registry
    (e.g. after a new skill has just been created).

    Returns a dict keyed by skill name:
    {
        "youtube-transcript": {
            "name": "youtube-transcript",
            "description": "...",
            "skill_md_path": Path(...),
            "scripts_dir": Path(...),
            "full_instructions": "...",
            "skill_dir": Path(...),
        },
        ...
    }
    """
    registry = {}

    if not SKILLS_DIR.exists():
        if not silent:
            print(f"[SkillsRegistry] WARNING: skills/ directory not found at {SKILLS_DIR}")
        return registry

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        content = skill_md.read_text(encoding="utf-8")
        meta = parse_frontmatter(content)

        name        = meta.get("name", skill_dir.name)
        description = meta.get("description", "No description available.")
        body        = meta.get("_body", content)
        scripts_dir = skill_dir / "scripts"

        registry[name] = {
            "name":             name,
            "description":      description,
            "skill_md_path":    skill_md,
            "scripts_dir":      scripts_dir if scripts_dir.exists() else None,
            "full_instructions": body,
            "skill_dir":        skill_dir,
        }

        if not silent:
            print(f"[SkillsRegistry] Loaded skill: '{name}'")

    if not silent:
        print(f"[SkillsRegistry] Total skills loaded: {len(registry)}")

    return registry


def get_registry(silent: bool = True) -> Dict[str, Dict]:
    """
    Always return a fresh registry read from disk.
    Use this everywhere instead of the old module-level REGISTRY constant.
    """
    return load_skill_registry(silent=silent)


def format_skills_for_prompt(registry: Dict[str, Dict]) -> str:
    """
    Format all skills as a prompt block for the LLM.
    Replicates the <available_skills> block Claude receives in its system prompt.
    """
    if not registry:
        return "No skills available."

    lines = ["## Available Skills\n"]
    for name, skill in registry.items():
        lines.append(f"### Skill: {name}")
        lines.append(f"**Description**: {skill['description']}\n")

    return "\n".join(lines)


def get_skill_instructions(registry: Dict[str, Dict], skill_name: str) -> Optional[str]:
    """
    Get the full instructions (SKILL.md body) for a given skill name.
    This is what Claude reads BEFORE executing a skill.
    """
    skill = registry.get(skill_name)
    if not skill:
        return None
    return skill["full_instructions"]

# def get_registry_summary(registry: dict[str, SkillMetadata]) -> str:
#     """
#     Format all skill name+description pairs for injection into the LLM context.
#     This is exactly what Claude sees in its <available_skills> block.
#     """
#     lines = ["<available_skills>"]
#     for name, meta in registry.items():
#         lines.append(f"""<skill>
# <name>{name}</name>
# <description>{meta.description}</description>
# </skill>""")
#     lines.append("</available_skills>")
#     return "\n".join(lines)

# ── Backwards-compatible singleton (refreshed on each import via get_registry) ─
REGISTRY = get_registry()

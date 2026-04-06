"""
skill_api_keys.py

Central registry of API keys required by each skill.
Maps skill-name → list of {env_var, label, help, required, is_password}.

app.py reads this at startup and renders input fields in the sidebar
for any key that is not already set in the environment.

Adding a new skill's keys:
  1. Add an entry to SKILL_API_KEYS below.
  2. The sidebar will auto-detect it when that skill is in the registry.
"""

from typing import Dict, List

# Each entry:
#   env_var     : os.environ key the script reads (e.g. os.getenv("SERPAPI_API_KEY"))
#   label       : Human-readable label shown in sidebar
#   help        : Tooltip / link shown under the input
#   required    : True = skill won't work without it, False = optional/fallback
#   is_password : True = masked input field
SKILL_API_KEYS: Dict[str, List[Dict]] = {

    # ── business-url-hybrid-search ────────────────────────────────────────────
    "browser-search": [
        {
            "env_var":     "SERPAPI_API_KEY",
            "label":       "SerpAPI Key",
            "help":        "Required — https://serpapi.com/manage-api-key",
            "required":    True,
            "is_password": True,
        },
       
    ],

    # ── Add future skills here ────────────────────────────────────────────────
    # "my-new-skill": [
    #     {
    #         "env_var":     "MY_API_KEY",
    #         "label":       "My Service API Key",
    #         "help":        "Get it at https://example.com/keys",
    #         "required":    True,
    #         "is_password": True,
    #     },
    # ],

}


def get_keys_for_skill(skill_name: str) -> List[Dict]:
    """Return the list of API key specs for a given skill (empty list if none)."""
    return SKILL_API_KEYS.get(skill_name, [])


def get_missing_keys(skill_name: str) -> List[Dict]:
    """Return only the required keys for a skill that are not yet set in os.environ."""
    import os
    missing = []
    for spec in get_keys_for_skill(skill_name):
        if spec["required"] and not os.environ.get(spec["env_var"], "").strip():
            missing.append(spec)
    return missing


def all_required_keys_present(skill_name: str) -> bool:
    """True if every required key for the skill is set in os.environ."""
    return len(get_missing_keys(skill_name)) == 0

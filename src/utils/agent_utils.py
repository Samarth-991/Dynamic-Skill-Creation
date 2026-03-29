# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE CONTENT EXTRACTION
# Gemini (via langchain-google-genai) can return content as:
#   - A plain str
#   - A list of dicts: [{"type": "text", "text": "..."}, ...]
#   - A complex object with __str__ that includes extras/signature/base64
# This function safely extracts ONLY the human-readable text in all cases.
# ══════════════════════════════════════════════════════════════════════════════
import os 
import sys
import re
import json
from typing import TypedDict, Annotated, Optional, List, Dict, Any


def extract_text_content(content: Any) -> str:
    """
    Safely extract plain text from any Gemini/LangChain content format.
    Never returns raw dicts, repr strings, or base64 blobs.
    """
    # Case 1: plain string
    if isinstance(content, str):
        text = content.strip()
        # Guard: if it looks like a raw Python repr or JSON object, clean it
        if text.startswith("[{") or text.startswith("{'"):
            # Try to parse and extract text fields
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return "\n".join(
                        b.get("text", "") for b in parsed
                        if isinstance(b, dict) and b.get("type") == "text"
                    ).strip()
            except Exception:
                pass
            # Fallback: strip the wrapping and find text= values
            matches = re.findall(r"'text':\s*'(.*?)'(?=\s*[,}])", text, re.DOTALL)
            if matches:
                return "\n".join(matches).strip()
        return text

    # Case 2: list of content blocks [{"type": "text", "text": "..."}, ...]
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
                elif "content" in block:
                    parts.append(str(block["content"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()

    # Case 3: object with a .text attribute (some LangChain wrappers)
    if hasattr(content, "text"):
        return str(content.text).strip()

    # Case 4: object with a .content attribute
    if hasattr(content, "content"):
        return extract_text_content(content.content)

    # Case 5: last resort — convert to string but strip anything after 'extras'
    raw = str(content)
    # Remove the extras/signature/base64 blob that Gemini appends
    for marker in ["', 'extras':", ", 'extras':", "extras=", "'extras':"]:
        if marker in raw:
            raw = raw[:raw.index(marker)]
    # Remove surrounding list/dict brackets if present
    raw = raw.strip("[]{}' \n")
    # If it starts with known field names, extract just the text value
    m = re.search(r"'text':\s*'(.*)", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip("'")
    return raw.strip()


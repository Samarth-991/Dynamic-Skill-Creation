from __future__ import annotations
import re
import logging
from typing import Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def remove_div_blocks(text: str) -> str:
    # Remove all <div ...>...</div> blocks (multiline)
    pattern = re.compile(r"<div.*?>.*?</div>", re.DOTALL)
    return re.sub(pattern, "", text)

def _strip_html(html: str) -> str:
    """
    Lightweight HTML → plain-text converter using stdlib only.
    Skips <script>, <style>, <nav>, <footer>, <header> blocks entirely.
    Falls back gracefully if BeautifulSoup is available.
    """
    try:
        from bs4 import BeautifulSoup          # prefer bs4 when installed
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        from html.parser import HTMLParser

        class _Extractor(HTMLParser):
            _SKIP_TAGS = {"script", "style", "nav", "footer", "header"}

            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
                self._skip_depth = 0

            def handle_starttag(self, tag, _attrs):
                if tag in self._SKIP_TAGS:
                    self._skip_depth += 1

            def handle_endtag(self, tag):
                if tag in self._SKIP_TAGS and self._skip_depth:
                    self._skip_depth -= 1

            def handle_data(self, data):
                if self._skip_depth == 0 and data.strip():
                    self.parts.append(data.strip())

        p = _Extractor()
        p.feed(html)
        text = "\n".join(p.parts)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── public API ────────────────────────────────────────────────────────────────

def read_markdown_file(uploaded_file) -> Tuple[str, str]:
    """
    Read content from a Streamlit UploadedFile object (.md / .txt / .rst).

    Args:
        uploaded_file : st.file_uploader result (must not be None).

    Returns:
        (content, source_hint)
        content     — full UTF-8 text of the file
        source_hint — human-readable label, e.g. "Uploaded file: SKILL.md"

    Raises:
        ValueError  — if the file cannot be decoded as UTF-8
        RuntimeError — if uploaded_file is None
    """
    if uploaded_file is None:
        raise RuntimeError("No file provided — uploaded_file is None.")

    try:
        raw_bytes = uploaded_file.read()
        content   = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Try latin-1 as a fallback before giving up
        try:
            content = raw_bytes.decode("latin-1")
            logger.warning(
                "File '%s' decoded with latin-1 fallback (not valid UTF-8).",
                uploaded_file.name,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot decode '{uploaded_file.name}' as UTF-8 or latin-1: {exc}"
            ) from exc

    source_hint = f"Uploaded file: {uploaded_file.name}"
    return content.strip(), source_hint


def fetch_url_content(url: str, timeout: int = 20) -> Tuple[str, str]:
    """
    Fetch text content from a URL.

    Handles:
      - Plain markdown / text (GitHub raw, docs sites with text/plain)
      - HTML pages              (stripped to readable prose)
      - GitHub blob URLs        → auto-converted to raw.githubusercontent.com

    Args:
        url     : Fully-qualified URL (must include https://).
        timeout : Request timeout in seconds (default 20).

    Returns:
        (content, source_hint)
        content     — extracted plain text / markdown
        source_hint — e.g. "github.com/owner/repo/blob/main/README.md"

    Raises:
        ImportError  — if the `requests` library is not installed
        ValueError   — if url is empty or not a valid http/https URL
        RuntimeError — on HTTP errors or unexpected content types
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "`requests` is required for URL fetching.  "
            "Install it with:  pip install requests"
        ) from exc

    url = url.strip()
    if not url:
        raise ValueError("url must not be empty.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Only http/https URLs are supported.  Got scheme: '{parsed.scheme}'"
        )

    if parsed.netloc in ("github.com", "www.github.com"):
        path_parts = parsed.path.lstrip("/").split("/")
        
        if len(path_parts) >= 4 and path_parts[2] == "blob":
            raw_path = "/".join(path_parts[:2] + path_parts[3:])
            url = f"https://raw.githubusercontent.com/{raw_path}"
            parsed = urlparse(url)
            logger.info("GitHub blob URL rewritten to raw: %s", url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SkillCreator-Bot/1.0; "
            "+https://github.com/your-org/skill-creator)"
        ),
        "Accept": "text/html,text/plain,text/markdown,application/xhtml+xml,*/*",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error fetching '{url}': {exc.response.status_code} "
            f"{exc.response.reason}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Connection error fetching '{url}': {exc}") from exc
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Request timed out after {timeout}s for URL: '{url}'"
        )

    content_type = resp.headers.get("content-type", "")

    if "text/html" in content_type:
        content = _strip_html(resp.text)
    else:
        # plain text / markdown / rst / xml — use as-is
        content = resp.text.strip()
    content = remove_div_blocks(content)  # clean up any leftover divs in HTML content
    source_hint = f"{parsed.netloc}{parsed.path}"
    return content,source_hint
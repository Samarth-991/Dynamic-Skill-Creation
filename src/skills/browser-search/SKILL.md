---
name: browser-search
description: >
  Use this skill whenever the user needs to find information on the internet,
  research a topic, look up current events, verify facts, or fetch and read
  the content of a specific URL. Triggers include: "search the web for",
  "find information about", "look up", "what is the latest on", "scrape this
  URL", "fetch this page", "research X", or any question that requires
  real-time or external data. Uses SerpAPI (via LangChain) to discover
  relevant URLs and Firecrawl (via LangChain) to fetch and convert page
  content into clean LLM-ready markdown.
---

# Browser Search Skill

A two-stage internet research pipeline:
1. **SerpAPI** — discovers relevant URLs from a Google search
2. **Firecrawl** — scrapes the top URL(s) and returns clean markdown

---

## Environment Variables

| Variable           | Required | Description                          |
|--------------------|----------|--------------------------------------|
| `SERPAPI_API_KEY`  | Yes      | From https://serpapi.com/manage-api-key |
| `FIRECRAWL_API_KEY`| Yes      | From https://www.firecrawl.dev/app/api-keys |

```bash
export SERPAPI_API_KEY="your_serpapi_key"
export FIRECRAWL_API_KEY="your_firecrawl_key"
```

---

## Installation

```bash
pip install google-search-results firecrawl-py langchain-community langchain-core
```

---

## Core Workflow

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Stage 1: SerpAPI Search    │  ← langchain_community.utilities.SerpAPIWrapper
│  Run query → top N results  │
│  Extract URLs + snippets    │
└────────────┬────────────────┘
             │  top URL(s)
             ▼
┌─────────────────────────────┐
│  Stage 2: Firecrawl Scrape  │  ← langchain_community.document_loaders.FireCrawlLoader
│  Fetch URL → clean markdown │
│  Returns: title, content,   │
│  sourceURL, metadata        │
└────────────┬────────────────┘
             │
             ▼
       Structured JSON Output
```

---

## Script: `browser_search.py`

```python
"""
browser_search.py
-----------------
Two-stage internet research:
  1. SerpAPI  → find relevant URLs for a query
  2. Firecrawl → scrape top URL(s) into clean markdown

Usage:
    python browser_search.py --query "your search query" [--top_n 1]

Output:
    JSON written to stdout (see Output Schema below)
"""

import os
import json
import argparse
from typing import Optional

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import FireCrawlLoader


# ── 1. SerpAPI: Search ────────────────────────────────────────────────────────

def serpapi_search(query: str, top_n: int = 3) -> list[dict]:
    """
    Run a Google search via SerpAPI and return the top N organic results.
    Each result has: title, url (link), snippet.

    The SerpAPIWrapper.results() method returns the full JSON response from
    SerpAPI. We extract the 'organic_results' list from it.

    Default params sent to SerpAPI:
        engine        = google
        google_domain = google.com
        gl            = us
        hl            = en
    Override via the `params` constructor argument if needed.
    """
    # API key is auto-read from SERPAPI_API_KEY env var
    search = SerpAPIWrapper()
    raw = search.results(query)                          # returns full dict
    organic = raw.get("organic_results", [])[:top_n]

    results = []
    for r in organic:
        results.append({
            "title":   r.get("title", ""),
            "url":     r.get("link", ""),
            "snippet": r.get("snippet", ""),
        })
    return results


# ── 2. Firecrawl: Scrape ──────────────────────────────────────────────────────

def firecrawl_scrape(url: str) -> Optional[dict]:
    """
    Scrape a single URL with Firecrawl and return its content as markdown.

    FireCrawlLoader modes:
        "scrape"  — single URL (used here)
        "crawl"   — all accessible subpages
        "map"     — URL discovery (returns list of semantically related links)
        "extract" — structured data extraction via LLM
        "search"  — Firecrawl's own web search

    Returns a dict with: title, content (markdown), sourceURL, metadata.
    API key is auto-read from FIRECRAWL_API_KEY env var.
    """
    loader = FireCrawlLoader(
        url=url,
        mode="scrape",
        params={
            "formats": ["markdown"],          # request clean markdown output
            "onlyMainContent": True,          # strip nav/footer boilerplate
        },
    )
    docs = loader.load()                      # returns list[Document]
    if not docs:
        return None

    doc = docs[0]
    return {
        "content":   doc.page_content,
        "title":     doc.metadata.get("title", ""),
        "sourceURL": doc.metadata.get("sourceURL", url),
        "metadata":  doc.metadata,
    }


# ── 3. Main pipeline ──────────────────────────────────────────────────────────

def browser_search(query: str, top_n: int = 1) -> dict:
    """
    Full two-stage pipeline:
        query → SerpAPI → top URL(s) → Firecrawl → structured result
    """
    # Stage 1: search
    try:
        search_results = serpapi_search(query, top_n=max(top_n, 3))
    except Exception as e:
        return {
            "status": "error",
            "url": None,
            "confidence": "low",
            "source": "serpapi",
            "metadata": {"error": str(e)},
        }

    if not search_results:
        return {
            "status": "not_found",
            "url": None,
            "confidence": "low",
            "source": "serpapi",
            "metadata": {"title": "", "snippet": "No results found for query."},
        }

    # Stage 2: scrape the top result
    top = search_results[0]
    url = top["url"]

    try:
        scraped = firecrawl_scrape(url)
    except Exception as e:
        # Firecrawl failed — return SerpAPI snippet as fallback
        return {
            "status": "success",
            "url": url,
            "confidence": "medium",
            "source": "serpapi",
            "metadata": {
                "title":   top["title"],
                "snippet": top["snippet"],
                "error":   f"Firecrawl scrape failed: {e}",
            },
        }

    if not scraped:
        return {
            "status": "not_found",
            "url": url,
            "confidence": "low",
            "source": "serpapi",
            "metadata": {"title": top["title"], "snippet": top["snippet"]},
        }

    return {
        "status":     "success",
        "url":        scraped["sourceURL"],
        "confidence": "high",
        "source":     "hybrid_match",          # SerpAPI found + Firecrawl confirmed
        "content":    scraped["content"],       # full markdown body
        "metadata": {
            "title":   scraped["title"] or top["title"],
            "snippet": top["snippet"],
        },
    }


# ── 4. CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browser search: SerpAPI + Firecrawl")
    parser.add_argument("--query",  required=True, help="Search query")
    parser.add_argument("--top_n",  type=int, default=1,
                        help="Number of top results to attempt scraping (default: 1)")
    args = parser.parse_args()

    result = browser_search(args.query, top_n=args.top_n)
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

## Output Schema

```json
{
  "status":     "success | not_found | error",
  "url":        "https://www.example.com",
  "confidence": "high | medium | low",
  "source":     "serpapi | firecrawl | hybrid_match",
  "content":    "Full markdown body of the scraped page (present on success)",
  "metadata": {
    "title":   "Page Title from SerpAPI / Firecrawl",
    "snippet": "Short description from SerpAPI organic result"
  }
}
```

| Field        | When present                                            |
|--------------|---------------------------------------------------------|
| `content`    | Only when Firecrawl successfully scraped the page       |
| `confidence` | `high` = full scrape; `medium` = snippet only; `low` = nothing found |
| `source`     | `hybrid_match` = SerpAPI + Firecrawl both succeeded     |

---

## Usage as a LangChain Agent Tool

Wrap the pipeline as a `@tool` so any LangChain / LangGraph agent can call it:

```python
from langchain_core.tools import tool

@tool
def browser_search_tool(query: str) -> str:
    """
    Search the internet for up-to-date information on any topic.
    Returns the full page content as markdown plus title and URL.
    Use when you need real-time facts, current events, or external data.
    """
    result = browser_search(query, top_n=1)
    if result["status"] != "success":
        return f"Search failed: {result['metadata'].get('snippet', 'No results')}"
    return (
        f"Source: {result['url']}\n"
        f"Title:  {result['metadata']['title']}\n\n"
        f"{result.get('content', result['metadata']['snippet'])}"
    )
```

### Wiring into an agent

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

llm   = ChatAnthropic(model="claude-sonnet-4-20250514")
agent = create_react_agent(llm, tools=[browser_search_tool])

response = agent.invoke({
    "messages": [{"role": "user", "content": "What are the latest advances in video diffusion models?"}]
})
print(response["messages"][-1].content)
```

---

## Firecrawl Modes Reference

| Mode      | Use case                                        | LangChain param         |
|-----------|-------------------------------------------------|-------------------------|
| `scrape`  | Single URL → markdown (default here)            | `mode="scrape"`         |
| `crawl`   | Entire site → all subpages                      | `mode="crawl"`          |
| `map`     | Discover all semantically related URLs          | `mode="map"`            |
| `extract` | Structured JSON extraction with LLM             | `mode="extract"`        |
| `search`  | Firecrawl's own web search (no SerpAPI needed)  | `mode="search"`         |

To scrape **multiple top results** in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

urls = [r["url"] for r in search_results[:3]]
with ThreadPoolExecutor(max_workers=3) as ex:
    scraped_pages = list(ex.map(firecrawl_scrape, urls))
```

---

## SerpAPIWrapper Params

The wrapper sends these defaults to the Google Search API. Override by passing
`params={}` to the constructor:

```python
search = SerpAPIWrapper(params={
    "engine":        "google",
    "google_domain": "google.com",
    "gl":            "in",     # country: India
    "hl":            "en",     # language
    "num":           10,       # number of results
})
```

Access raw results (full JSON) with `search.results(query)`, or a single
summarised string with `search.run(query)`.

---

## Error Handling & Confidence Levels

| Scenario                              | `status`    | `confidence` | `source`      |
|---------------------------------------|-------------|--------------|---------------|
| SerpAPI + Firecrawl both succeed      | `success`   | `high`       | `hybrid_match`|
| SerpAPI ok, Firecrawl fails           | `success`   | `medium`     | `serpapi`     |
| SerpAPI returns no results            | `not_found` | `low`        | `serpapi`     |
| SerpAPI API call throws exception     | `error`     | `low`        | `serpapi`     |

---

## Common Pitfalls

- **`SERPAPI_API_KEY` not set** → `ValueError` from `SerpAPIWrapper` on init.
- **`FIRECRAWL_API_KEY` not set** → `ValueError` from `FireCrawlLoader` on init.
- **JavaScript-heavy pages** → Firecrawl handles JS rendering automatically; no extra config needed.
- **Paywalled / auth-gated pages** → Firecrawl cannot bypass login walls. Fall back to SerpAPI snippet.
- **Rate limits** → SerpAPI free tier: 100 searches/month. Firecrawl free tier: 500 credits. Cache results for repeated queries.
- **`onlyMainContent=True`** strips navigation and footer noise — recommended for LLM consumption.
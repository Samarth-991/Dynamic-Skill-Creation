"""
browser_search.py
-----------------
Two-stage internet research pipeline:
  1. SerpAPI    → search Google and return top organic results (title, url, snippet)
  2. Firecrawl  → scrape the top URL(s) into clean LLM-ready markdown

Requirements:
    pip install -r requirements.txt

Environment variables:
    SERPAPI_API_KEY   — from https://serpapi.com/manage-api-key
    FIRECRAWL_API_KEY — from https://www.firecrawl.dev/app/api-keys

Usage (CLI):
    python browser_search.py --query "latest advances in video diffusion models"
    python browser_search.py --query "Anthropic Claude 4 release date" --top_n 3

Usage (as a LangChain agent tool):
    from browser_search import browser_search_tool
    agent = create_react_agent(llm, tools=[browser_search_tool])
"""

import os
import json
import argparse
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import FireCrawlLoader
from langchain_core.tools import tool


# ── Validation ────────────────────────────────────────────────────────────────

def _check_env():
    missing = [k for k in ("SERPAPI_API_KEY", "FIRECRAWL_API_KEY") if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            "Set them before running:\n"
            "  export SERPAPI_API_KEY=your_key\n"
            "  export FIRECRAWL_API_KEY=your_key"
        )


# ── Stage 1: SerpAPI Search ───────────────────────────────────────────────────

def serpapi_search(query: str, top_n: int = 3) -> list[dict]:
    """
    Query Google via SerpAPI and return the top N organic results.

    Each result dict contains:
        title   — page title
        url     — canonical link
        snippet — short description from the SERP

    SerpAPIWrapper default params:
        engine        = google
        google_domain = google.com
        gl            = us  (country)
        hl            = en  (language)

    Override by passing params={} to SerpAPIWrapper constructor, e.g.:
        SerpAPIWrapper(params={"gl": "in", "hl": "en", "num": 10})

    API key is auto-read from SERPAPI_API_KEY environment variable.
    """
    search = SerpAPIWrapper()
    raw = search.results(query)                     # full JSON response from SerpAPI
    organic = raw.get("organic_results", [])[:top_n]

    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("link", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in organic
    ]


# ── Stage 2: Firecrawl Scrape ─────────────────────────────────────────────────

def firecrawl_scrape(url: str) -> Optional[dict]:
    """
    Scrape a single URL with Firecrawl and return structured content.

    FireCrawlLoader modes (set via mode= argument):
        "scrape"   — single URL → markdown (used here, default)
        "crawl"    — entire site → all accessible subpages
        "map"      — URL discovery, returns list of related links
        "extract"  — structured JSON extraction via LLM
        "search"   — Firecrawl's own web search (no SerpAPI needed)

    Returns:
        {
            "content":   str   — full page content as clean markdown
            "title":     str   — page title
            "sourceURL": str   — canonical URL of the scraped page
            "metadata":  dict  — raw metadata from Firecrawl
        }
        or None if scraping returned no documents.

    API key is auto-read from FIRECRAWL_API_KEY environment variable.
    """
    loader = FireCrawlLoader(
        url=url,
        mode="scrape",
        params={
            "formats":         ["markdown"],   # return clean markdown
            "onlyMainContent": True,           # strip nav, footer, sidebar boilerplate
        },
    )
    docs = loader.load()                       # returns list[Document]
    if not docs:
        return None

    doc = docs[0]
    return {
        "content":   doc.page_content,
        "title":     doc.metadata.get("title", ""),
        "sourceURL": doc.metadata.get("sourceURL", url),
        "metadata":  doc.metadata,
    }


def firecrawl_scrape_many(urls: list[str], max_workers: int = 3) -> list[Optional[dict]]:
    """
    Scrape multiple URLs in parallel using a thread pool.

    Args:
        urls        — list of URLs to scrape
        max_workers — number of parallel threads (default: 3)

    Returns a list of results in the same order as the input URLs.
    Failed scrapes are returned as None (no exception raised).
    """
    def _safe_scrape(url):
        try:
            return firecrawl_scrape(url)
        except Exception as e:
            return {"error": str(e), "url": url}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_safe_scrape, urls))


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_browser_search(query: str, top_n: int = 1) -> dict:
    """
    Full two-stage pipeline: query → SerpAPI → Firecrawl → structured result.

    Args:
        query  — natural language search query
        top_n  — number of top SerpAPI results to attempt scraping (default: 1)

    Returns a dict matching the Output Schema:
        {
            "status":     "success" | "not_found" | "error"
            "url":        str | None
            "confidence": "high" | "medium" | "low"
            "source":     "serpapi" | "firecrawl" | "hybrid_match"
            "content":    str   (only on success with Firecrawl)
            "metadata": {
                "title":   str,
                "snippet": str
            }
        }

    Confidence levels:
        high   — SerpAPI + Firecrawl both succeeded (hybrid_match)
        medium — SerpAPI ok, Firecrawl failed; snippet returned as fallback
        low    — No results found, or SerpAPI itself threw an exception
    """
    # ── Stage 1: Search ───────────────────────────────────────────────────────
    try:
        search_results = serpapi_search(query, top_n=max(top_n, 3))
    except Exception as e:
        return {
            "status":     "error",
            "url":        None,
            "confidence": "low",
            "source":     "serpapi",
            "metadata":   {"error": str(e)},
        }

    if not search_results:
        return {
            "status":     "not_found",
            "url":        None,
            "confidence": "low",
            "source":     "serpapi",
            "metadata":   {"title": "", "snippet": "No results found for query."},
        }

    # ── Stage 2: Scrape top result ────────────────────────────────────────────
    top = search_results[0]
    url = top["url"]

    try:
        scraped = firecrawl_scrape(url)
    except Exception as e:
        # Firecrawl failed — degrade gracefully to SerpAPI snippet
        return {
            "status":     "success",
            "url":        url,
            "confidence": "medium",
            "source":     "serpapi",
            "metadata": {
                "title":   top["title"],
                "snippet": top["snippet"],
                "error":   f"Firecrawl scrape failed: {e}",
            },
        }

    if not scraped:
        return {
            "status":     "not_found",
            "url":        url,
            "confidence": "low",
            "source":     "serpapi",
            "metadata":   {"title": top["title"], "snippet": top["snippet"]},
        }

    return {
        "status":     "success",
        "url":        scraped["sourceURL"],
        "confidence": "high",
        "source":     "hybrid_match",
        "content":    scraped["content"],
        "metadata": {
            "title":   scraped["title"] or top["title"],
            "snippet": top["snippet"],
        },
    }


# ── LangChain Agent Tool ──────────────────────────────────────────────────────

@tool
def browser_search_tool(query: str) -> str:
    """
    Search the internet for up-to-date information on any topic.

    Performs a Google search via SerpAPI, then scrapes the top result with
    Firecrawl to return full page content as clean markdown.

    Use this tool when you need:
      - Real-time facts or current events
      - Information beyond your training cutoff
      - Content from a specific website or article

    Args:
        query — natural language search query

    Returns:
        A formatted string with source URL, title, and full page content.
        Falls back to the SerpAPI snippet if scraping fails.
    """
    result = run_browser_search(query, top_n=1)

    if result["status"] == "error":
        return f"Search error: {result['metadata'].get('error', 'Unknown error')}"

    if result["status"] == "not_found":
        return f"No results found for: {query}"

    content = result.get("content") or result["metadata"].get("snippet", "No content available.")

    return (
        f"Source:     {result['url']}\n"
        f"Title:      {result['metadata']['title']}\n"
        f"Confidence: {result['confidence']}\n\n"
        f"{content}"
    )


# ── Agent Example ─────────────────────────────────────────────────────────────

def run_agent_example(query: str):
    """
    Example: plug browser_search_tool into a LangGraph ReAct agent backed by Claude.
    Requires: pip install langchain-anthropic langgraph
    """
    from langchain_groq import ChatGroq
    from langchain.agents import create_agent
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="openai/gpt-oss-120b")
    agent = create_agent(llm, tools=[browser_search_tool])

    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    print(response["messages"][-1].content)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browser search using SerpAPI + Firecrawl via LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python browser_search.py --query "what is LangGraph"
  python browser_search.py --query "Anthropic Claude 4 features" --top_n 3
  python browser_search.py --query "best CV papers 2025" --agent
        """
    )
    parser.add_argument("--query",  required=True,      help="Search query")
    parser.add_argument("--top_n",  type=int, default=1, help="Top N results to scrape (default: 1)")
    parser.add_argument("--agent",  action="store_true", help="Run via LangGraph ReAct agent instead")
    args = parser.parse_args()

    _check_env()

    if args.agent:
        run_agent_example(args.query)
    else:
        result = browser_search(args.query, top_n=args.top_n)
        print(json.dumps(result, indent=2, ensure_ascii=False))
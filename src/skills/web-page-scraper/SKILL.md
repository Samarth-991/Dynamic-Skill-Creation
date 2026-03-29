---
name: web-page-scraper
description: Trigger this skill IMMEDIATELY when the user asks to "scrape", "read", "extract text", or "get content" from a specific URL or a general search topic. Use this for requests like "what does website X say", "summarize this link", or "find and read a page about Y". It handles both direct URL access and search-based discovery to return structured text data.
---

# Web Page Scraper

## Overview
The `web-page-scraper` is a specialized utility designed to retrieve and parse human-readable content from the web. It operates in two modes: Direct Access (scraping a specific URL provided by the user) and Discovery (searching via DuckDuckGo to find a relevant URL, then scraping it). The output is strictly structured JSON, making it ideal for downstream summarization or analysis tasks.

## Automatic Processing
*   **Input Detection:** The system automatically distinguishes between a valid HTTP/HTTPS URL and a natural language search query.
*   **Header Management:** Automatically rotates User-Agent headers to mimic a standard browser and minimize 403 Forbidden errors.
*   **Content Cleaning:** Automatically strips HTML tags, `<script>`, `<style>`, and navigation elements to isolate the main body text.

## Core Capabilities
1.  **URL Scraping:** Fetches HTML from a target URL and parses the DOM.
2.  **Search Integration:** Uses `duckduckgo-search` to resolve queries into scrape-able URLs.
3.  **Structure Extraction:** Separates page metadata (Title) from structural elements (Headers) and body content (Paragraphs).
4.  **Error Resilience:** Handles HTTP errors and connection timeouts gracefully.

## Workflow

### Step 1: Input Analysis
Determine if the user input is a direct URL or a search query.
*   If input starts with `http://` or `https://`, treat as **Direct URL**.
*   Otherwise, treat as **Search Query**.

### Step 2: Execution (Python Script)
Execute the following Python script. Pass the user input as the variable `target_input`.

```python
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import json
import sys
import re

# CONFIGURATION
target_input = """{{target_input}}"""  # Input from user
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_url_from_query(query):
    try:
        results = list(DDGS().text(query, max_results=1))
        if results:
            return results[0]['href']
        return None
    except Exception as e:
        return None

def scrape_url(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.extract()

        # Extract Title
        title = soup.title.string if soup.title else "No Title"
        
        # Extract Headers
        headers = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
        
        # Extract Text
        text = soup.get_text(separator=' ')
        
        # Clean Text (remove excess whitespace)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return {
            "status": "success",
            "source_url": url,
            "title": title.strip(),
            "headers": headers[:10], # Top 10 headers
            "content": clean_text[:5000] # Limit to first 5000 chars to prevent overflow
        }
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "message": f"HTTP Error: {e.response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    url_pattern = re.compile(r'^https?://')
    
    if url_pattern.match(target_input):
        final_url = target_input
    else:
        final_url = get_url_from_query(target_input)
        
    if not final_url:
        print(json.dumps({"status": "error", "message": "Could not find a valid URL for the query."}))
        return

    data = scrape_url(final_url)
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
```

### Step 3: Output Validation
1.  Check if the script output is valid JSON.
2.  Check the `status` field.
    *   If `success`: Proceed to formatting.
    *   If `error`: Report the error message to the user.

### Step 4: Final Output Generation
Return the raw JSON object. Do not summarize unless explicitly asked by the user in the prompt *after* the skill execution.

## Usage Patterns

### Pattern 1: Direct URL Scraping
**User:** "Scrape the content from https://www.python.org"
**Action:** Script detects URL -> Fetches HTML -> Returns JSON.

### Pattern 2: Search and Scrape
**User:** "Find a recipe for apple pie and get the text."
**Action:** Script detects query -> DDGS finds top result -> Scrapes that URL -> Returns JSON.

### Pattern 3: Contextual Reading
**User:** "Read this page: https://example.com/article"
**Action:** Script scrapes URL -> Returns JSON -> (Agent uses JSON to answer questions).

## Error Handling

| Error Type | Likely Cause | User Message |
| :--- | :--- | :--- |
| `HTTP Error: 403` | Anti-bot protection or firewall. | "I cannot access this website due to security restrictions (403 Forbidden)." |
| `HTTP Error: 404` | Page does not exist. | "The requested page could not be found (404)." |
| `Connection Timeout` | Server is slow or down. | "The website took too long to respond." |
| `No URL Found` | Search query yielded no results. | "I couldn't find a relevant website for that search query." |
| `JSON Decode Error` | Script failed to output JSON. | "An internal error occurred while processing the page content." |

## Output Formatting

The output is always a JSON object with the following schema:

```json
{
  "status": "success",
  "source_url": "https://...",
  "title": "Page Title",
  "headers": ["Header 1", "Header 2"],
  "content": "Full body text content..."
}
```

*   **Short Content:** If content is < 500 chars, display fully.
*   **Long Content:** The script truncates at 5000 chars. If the user needs more, suggest they visit the URL directly.

## Best Practices
1.  **Respect Privacy:** Do not attempt to scrape login-walled or password-protected pages.
2.  **Rate Limiting:** Do not trigger this skill in a rapid loop against the same domain.
3.  **Attribution:** Always ensure the `source_url` is visible in the final response so the user knows where the data came from.
4.  **Relevance:** If the search query is vague, the scraper picks the first result. Encourage users to be specific (e.g., "python documentation" vs "python").
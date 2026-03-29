import sys
import json
import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

def is_url(input_value: str) -> bool:
    """
    Checks if the input string looks like a valid URL.
    Requires http:// or https:// prefix to be considered a direct URL.
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, input_value) is not None

def get_url_from_query(query: str) -> str:
    """
    Uses DuckDuckGo to find the first URL for a search query.
    Returns None if no results are found.
    """
    try:
        with DDGS() as ddgs:
            # Fetch 1 result
            results = list(ddgs.text(query, max_results=1))
            if results:
                return results[0]['href']
    except Exception as e:
        raise Exception(f"DuckDuckGo search failed: {str(e)}")
    return None

def scrape_page(url: str) -> dict:
    """
    Fetches and parses the HTML content of a URL.
    Extracts title, headers (h1-h3), and main body text.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch URL: {str(e)}")

    try:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract Title
        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"

        # Extract Headers (h1, h2, h3)
        headers_data = []
        for tag in soup.find_all(['h1', 'h2', 'h3']):
            text = tag.get_text(strip=True)
            if text:
                headers_data.append({
                    'level': tag.name,
                    'text': text
                })

        # Extract Body Text
        # Remove unwanted tags to reduce noise
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'meta', 'noscript', 'iframe', 'svg']):
            element.decompose()

        # Get text with separator to prevent words merging
        body_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace
        body_text = re.sub(r'\s+', ' ', body_text)

        return {
            "url": url,
            "title": title,
            "headers": headers_data,
            "body_text": body_text
        }
    except Exception as e:
        raise Exception(f"Error parsing HTML content: {str(e)}")

def run_web_page_scraper(input_value: str) -> dict:
    """
    Main entry point for the skill.
    1. Identifies if input is a URL or a search query.
    2. Resolves a URL if a query is provided.
    3. Scrapes the target URL.
    4. Returns a dictionary with success status and data.
    """
    try:
        target_url = input_value.strip()
        
        # Determine if input is a URL or a search query
        if not is_url(target_url):
            # It's a query, search for it
            found_url = get_url_from_query(target_url)
            if not found_url:
                return {
                    "success": False,
                    "error": f"No search results found for query: '{target_url}'"
                }
            target_url = found_url

        # Scrape the URL
        scraped_data = scrape_page(target_url)
        
        return {
            "success": True,
            "data": scraped_data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # CLI Entry Point
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False, 
            "error": "No input provided. Usage: python web_page_scraper.py <url_or_query>"
        }))
        sys.exit(1)
    
    # Join arguments in case the query was not quoted
    input_arg = " ".join(sys.argv[1:])
    result = run_web_page_scraper(input_arg)
    print(json.dumps(result, indent=2))
import sys
import json
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
UTILS_DIR   = PROJECT_ROOT / "utils/"
sys.path.append(str(UTILS_DIR))
from utils import content_reader


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

def scrape_page(url: str) -> dict:
    """
    Scrapes the given URL and extracts the page title, headers, and main text content.
    Returns a dictionary with the extracted information.
    """
    try:
        url_content= content_reader.fetch_url_content(url)  # This will raise an exception if the URL is not reachable or valid 
        return url_content
    except Exception as e:
        print (Exception(f"Failed to fetch content from URL '{url}': {str(e)}"))
        return {
           "success": False,
            "error": f"Failed to fetch content from URL '{url}': {str(e)}"
        }


def run_web_page_scraper(url: str) -> dict:
    """
    Main function to run the web page scraper skill.
    This function is used to only scrape the webpage and return the content. It does not perform search-based discovery. 
    If the input is a query, it will first find the URL using DuckDuckGo and then scrape it.
    """
    try:
        target_url = url.strip()
        # Determine if input is a URL or a search query
        if is_url(target_url):
           scraped_data =  scrape_page(target_url)  # Just to check if the URL is reachable and valid
        else:
            return {
                    "success": False,
                    "error": f"Please provide the url: '{target_url}'"
                }
        
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
    print("Web Page Scraper Skill - CLI Mode")
    # CLI Entry Point
    # if len(sys.argv) < 2:
    #     print(json.dumps({
    #         "success": False, 
    #         "error": "No input provided. Usage: python web_page_scraper.py <url_or_query>"
    #     }))
    #     sys.exit(1)
    
    # # Join arguments in case the query was not quoted
    # input_arg = " ".join(sys.argv[1:])
    # result = run_web_page_scraper(input_arg)
    # print(json.dumps(result, indent=2))
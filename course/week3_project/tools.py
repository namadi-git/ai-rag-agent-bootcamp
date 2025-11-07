from dotenv import load_dotenv
load_dotenv()

import requests
from dataclasses import dataclass
from typing import List
from html import unescape
import requests
from dataclasses import dataclass

@dataclass
class SearchQuery:
    """
    A single Wikipedia search hit.

    Attributes:
        title:   Page title.
        snippet: Short snippet (plain text) highlighting the match.
    """
    title: str
    snippet: str


@dataclass
class WebPageContent:
    """
    Container for fetched Wikipedia page content.

    Attributes:
        wiki_url:  Canonical page URL.
        title:     Page title.
        content:   Extracted plain-text intro (may be empty on failure).
    """
    wiki_url: str
    title: str
    content: str


def get_search_queries(subject: str) -> List[SearchQuery]:
    """Return titles/snippets for a Wikipedia search query. 
    Args:
        subject: Search query string.
    Returns:
        List of SearchQuery objects with title and snippet.
    """
    WIKI_API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "YourApp/0.1 (you@example.com)"}
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": subject,
        "srlimit": 10,
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()  # <-- real JSON, no second json.loads needed

    results: List[SearchQuery] = []
    for item in data.get("query", {}).get("search", []):
        snippet_html = item.get("snippet", "")
        # strip the tiny bit of HTML Wikipedia includes in snippets
        snippet = (snippet_html
                   .replace('<span class="searchmatch">', "")
                   .replace("</span>", ""))
        results.append(SearchQuery(title=item.get("title", ""), snippet=unescape(snippet).strip()))
    return results
    



def fetch_wiki_page(title: str) -> WebPageContent:
    """
    Fetch a page’s plain-text introduction from Wikipedia.

    Args:
        title: Exact or close page title (e.g., "Capybara").

    Returns:
        WebPageContent with canonical URL, normalized title, and plain-text extract.
        On failure, returns an 'Error' container with a brief message.
    """
    WIKI_API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "YourApp/0.1 (you@example.com)"}
    try:
        # Use extracts API to get a clean intro paragraph as plain text.
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()  # <-- real JSON, no second json.loads needed
        pages = data.get("query", {}).get("pages", {})

        if not pages:
            return WebPageContent(
                wiki_url="N/A",
                title="Error",
                content=f"No page found for: {title}",
            )

        # pages is a dict keyed by pageid; pick the first
        page = next(iter(pages.values()))
        normalized_title = page.get("title", title)
        extract = page.get("extract", "") or ""

        canonical_url = f"https://en.wikipedia.org/wiki/{normalized_title.replace(' ', '_')}"

        return WebPageContent(
            wiki_url=canonical_url,
            title=normalized_title,
            content=extract.strip(),
        )
    except Exception:
        return WebPageContent(
            wiki_url="N/A",
            title="Error",
            content=f"Error fetching page for: {title}",
        )
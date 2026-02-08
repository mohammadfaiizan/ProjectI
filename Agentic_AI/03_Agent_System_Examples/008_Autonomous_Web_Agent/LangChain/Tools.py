"""
Tools module for Autonomous Web Agent.
Contains web scraping tools, content parsing, and caching utilities.
"""

from langchain_core.tools import tool
from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import hashlib
from collections import defaultdict


class Web_Browser:
    """Session-based HTTP client with headers and cookie handling."""
    
    def __init__(
        self,
        timeout: int = 30,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        follow_redirects: bool = True,
        verify_ssl: bool = True
    ):
        """
        Initialize web browser session.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: User agent string
            follow_redirects: Whether to follow redirects
            verify_ssl: Whether to verify SSL certificates
        """
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        })
    
    def Fetch(self, url: str) -> Tuple[Optional[str], Optional[str], int]:
        """
        Fetch a URL and return content, content type, and status code.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (content, content_type, status_code)
        """
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=self.follow_redirects,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            
            if "text/html" in content_type or "text/plain" in content_type:
                return response.text, content_type, response.status_code
            else:
                return None, content_type, response.status_code
        except requests.exceptions.RequestException as e:
            return None, None, 0
    
    def Close(self):
        """Close the session."""
        self.session.close()


class Content_Parser:
    """Class for cleaning HTML and extracting main content."""
    
    def __init__(self, max_content_length: int = 100000):
        """
        Initialize content parser.
        
        Args:
            max_content_length: Maximum content length to process
        """
        self.max_content_length = max_content_length
    
    def Clean_HTML(self, html_content: str) -> str:
        """
        Clean HTML content by removing scripts, styles, and other non-content elements.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Remove navigation and footer elements
        for element in soup.find_all(["nav", "footer", "header"]):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator="\n", strip=True)
        
        # Truncate if too long
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "... [truncated]"
        
        return text
    
    def Extract_Main_Content(self, html_content: str) -> str:
        """
        Extract main content from HTML, prioritizing article/main tags.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Main content text
        """
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Try to find main content areas
        main_content = None
        for tag in ["main", "article", "[role='main']", ".content", "#content"]:
            element = soup.select_one(tag)
            if element:
                main_content = element
                break
        
        if not main_content:
            # Fallback to body
            main_content = soup.find("body") or soup
        
        # Remove unwanted elements
        for element in main_content.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        text = main_content.get_text(separator="\n", strip=True)
        
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "... [truncated]"
        
        return text
    
    def Extract_Links_From_HTML(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """
        Extract all links from HTML with their text.
        
        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of dictionaries with 'url' and 'text' keys
        """
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "")
            text = anchor.get_text(strip=True)
            
            if href:
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                # Only include HTTP/HTTPS links
                parsed = urlparse(absolute_url)
                if parsed.scheme in ["http", "https"]:
                    links.append({
                        "url": absolute_url,
                        "text": text or href
                    })
        
        return links
    
    def Extract_Tables_From_HTML(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Extract HTML tables as structured data.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            List of dictionaries with table data
        """
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, "html.parser")
        tables = []
        
        for table in soup.find_all("table"):
            table_data = {
                "headers": [],
                "rows": []
            }
            
            # Extract headers
            header_row = table.find("thead")
            if header_row:
                headers = header_row.find_all(["th", "td"])
                table_data["headers"] = [h.get_text(strip=True) for h in headers]
            else:
                # Try first row as headers
                first_row = table.find("tr")
                if first_row:
                    headers = first_row.find_all(["th", "td"])
                    table_data["headers"] = [h.get_text(strip=True) for h in headers]
            
            # Extract rows
            tbody = table.find("tbody") or table
            for row in tbody.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    table_data["rows"].append(row_data)
            
            if table_data["rows"]:
                tables.append(table_data)
        
        return tables


class Page_Cache:
    """Cache for fetched pages to avoid re-fetching."""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize page cache.
        
        Args:
            ttl: Time to live in seconds (default 1 hour)
        """
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl = ttl
    
    def Get(self, url: str) -> Optional[str]:
        """
        Get cached content for a URL.
        
        Args:
            url: URL to look up
            
        Returns:
            Cached content or None if not found/expired
        """
        if url not in self.cache:
            return None
        
        content, timestamp = self.cache[url]
        
        if time.time() - timestamp > self.ttl:
            del self.cache[url]
            return None
        
        return content
    
    def Set(self, url: str, content: str):
        """
        Cache content for a URL.
        
        Args:
            url: URL to cache
            content: Content to cache
        """
        self.cache[url] = (content, time.time())
    
    def Clear(self):
        """Clear all cached content."""
        self.cache.clear()
    
    def Get_Cache_Size(self) -> int:
        """Return number of cached pages."""
        return len(self.cache)


# Global instances
_browser = None
_parser = None
_cache = None


def Get_Browser() -> Web_Browser:
    """Get or create global browser instance."""
    global _browser
    if _browser is None:
        _browser = Web_Browser()
    return _browser


def Get_Parser() -> Content_Parser:
    """Get or create global parser instance."""
    global _parser
    if _parser is None:
        _parser = Content_Parser()
    return _parser


def Get_Cache() -> Page_Cache:
    """Get or create global cache instance."""
    global _cache
    if _cache is None:
        _cache = Page_Cache()
    return _cache


@tool
def Fetch_Web_Page(url: str) -> str:
    """
    Fetch a web page and return its text content.
    Uses BeautifulSoup to parse HTML and extract readable text.
    Includes mock fallback if requests fails.
    
    Args:
        url: URL of the web page to fetch
        
    Returns:
        Text content of the web page
    """
    cache = Get_Cache()
    cached_content = cache.Get(url)
    if cached_content:
        return cached_content
    
    browser = Get_Browser()
    content, content_type, status_code = browser.Fetch(url)
    
    if content and status_code == 200:
        parser = Get_Parser()
        cleaned_content = parser.Extract_Main_Content(content)
        cache.Set(url, cleaned_content)
        return cleaned_content
    else:
        # Mock fallback for demonstration
        return f"[Mock Content] Page content for {url} would be fetched here. Status: {status_code}"


@tool
def Extract_Links(url: str) -> List[Dict[str, str]]:
    """
    Fetch a web page and extract all links with their text.
    
    Args:
        url: URL of the web page to extract links from
        
    Returns:
        List of dictionaries with 'url' and 'text' keys
    """
    browser = Get_Browser()
    content, _, status_code = browser.Fetch(url)
    
    if content and status_code == 200:
        parser = Get_Parser()
        links = parser.Extract_Links_From_HTML(content, url)
        return links
    else:
        return []


@tool
def Extract_Tables(url: str) -> List[Dict[str, Any]]:
    """
    Extract HTML tables from a web page as structured data.
    
    Args:
        url: URL of the web page to extract tables from
        
    Returns:
        List of dictionaries containing table headers and rows
    """
    browser = Get_Browser()
    content, _, status_code = browser.Fetch(url)
    
    if content and status_code == 200:
        parser = Get_Parser()
        tables = parser.Extract_Tables_From_HTML(content)
        return tables
    else:
        return []


@tool
def Search_Page_Content(url: str, query: str) -> str:
    """
    Search within a page's content for specific information.
    Returns relevant excerpts matching the query.
    
    Args:
        url: URL of the web page to search
        query: Search query string
        
    Returns:
        Relevant excerpts from the page matching the query
    """
    cache = Get_Cache()
    cached_content = cache.Get(url)
    
    if not cached_content:
        browser = Get_Browser()
        content, _, status_code = browser.Fetch(url)
        
        if content and status_code == 200:
            parser = Get_Parser()
            cached_content = parser.Extract_Main_Content(content)
            cache.Set(url, cached_content)
        else:
            return f"Could not fetch page: {url}"
    
    # Simple keyword-based search
    query_lower = query.lower()
    lines = cached_content.split("\n")
    matching_lines = []
    
    for line in lines:
        if query_lower in line.lower():
            matching_lines.append(line.strip())
            if len(matching_lines) >= 10:  # Limit results
                break
    
    if matching_lines:
        return "\n".join(matching_lines)
    else:
        return f"No content found matching '{query}' on page {url}"

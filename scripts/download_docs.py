"""
Script to download LangChain documentation for RAG corpus.
Downloads from LangChain Python API docs and tutorials.
"""
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

# Base URLs for LangChain documentation
BASE_URLS = [
    "https://python.langchain.com/api_reference/",
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/how_to/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

OUTPUT_DIR = "data/raw"


def get_links_from_page(url: str, visited: set) -> list:
    """Extract documentation links from a page."""
    links = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(url, href)
                # Only include langchain docs
                if 'python.langchain.com' in full_url and full_url not in visited:
                    if '/api_reference/' in full_url or '/docs/' in full_url:
                        links.append(full_url)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return links


def extract_content(url: str) -> dict:
    """Extract text content from a documentation page."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else urlparse(url).path
            
            # Get main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            return {
                "url": url,
                "title": title_text,
                "content": clean_text[:50000]  # Limit content size
            }
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    visited = set()
    to_visit = list(BASE_URLS)
    documents = []
    
    print("Starting documentation download...")
    
    # Crawl pages (limit to 100 pages for reasonable corpus size)
    max_pages = 100
    pbar = tqdm(total=max_pages, desc="Downloading docs")
    
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        
        visited.add(url)
        
        # Extract content
        doc = extract_content(url)
        if doc and len(doc['content']) > 100:  # Skip very short pages
            documents.append(doc)
            pbar.update(1)
        
        # Get more links from this page
        new_links = get_links_from_page(url, visited)
        to_visit.extend(new_links[:10])  # Add up to 10 new links per page
        
        time.sleep(0.5)  # Be nice to the server
    
    pbar.close()
    
    # Save documents
    output_file = os.path.join(OUTPUT_DIR, "langchain_docs.json")
    with open(output_file, 'w') as f:
        json.dump(documents, f, indent=2)
    
    print(f"\nDownloaded {len(documents)} documents")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()


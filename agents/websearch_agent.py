import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# 1. Tavily Configuration
#    Set your token in an environment variable or hardcode for convenience.
# ---------------------------------------------------------------------------
TAVILY_TOKEN = os.getenv("TAVILY_API_KEY")  
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


# ---------------------------------------------------------------------------
# 2. Tavily Search Function
#    Uses POST /search to get relevant links and short snippet/answer
# ---------------------------------------------------------------------------
def tavily_search(query,
                  topic="general",
                  search_depth="advanced",
                  max_results=3,
                  days=7,
                  include_domains=None,
                  exclude_domains=None,
                  location=None):
    """
    Perform a Tavily search and return search results JSON.
    """
    if location and location not in query:
        enhanced_query = f"{query} in {location}"
    else:
        enhanced_query = query
    payload = {
        "query": enhanced_query,
        "topic": topic,               # e.g., "news", "general", ...
        "search_depth": search_depth, # "basic", "advanced"
        "chunks_per_source": 3,
        "max_results": max_results,
        "days": days,                 # past N days for filtering
        "include_answer": True,
        "include_raw_content": False,
        "include_images": True,
        "include_image_descriptions": True,
        "include_domains": include_domains or [],
        "exclude_domains": exclude_domains or []
    }
    headers = {
        "Authorization": f"Bearer {TAVILY_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(TAVILY_SEARCH_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[Error] Tavily search failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Tavily Extract Function
#    Uses POST /extract to get detailed content from URLs
# ---------------------------------------------------------------------------
def tavily_extract(urls, extract_depth="basic", include_images=True):  # Default to True
    """
    Extract content (text, optional images) from a list of URLs.
    """
    if isinstance(urls, str):
        # convert single URL to a list for consistency
        urls = [urls]

    payload = {
        "urls": urls,
        "extract_depth": extract_depth,
        "include_images": include_images
    }
    headers = {
        "Authorization": f"Bearer {TAVILY_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(TAVILY_EXTRACT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[Error] Tavily extract failed: {e}")
        return None

# Simple function to get fallback images if API doesn't return any
def get_fallback_images():
    """Return fallback crime-related image URLs"""
    return [
        "https://source.unsplash.com/featured/?police",
        "https://source.unsplash.com/featured/?crime,scene",
        "https://source.unsplash.com/featured/?police,car"
    ]

# ---------------------------------------------------------------------------
# 4. Generate Markdown
#    Build a neat Markdown output from the search + extraction results.
# ---------------------------------------------------------------------------
def build_markdown_report(query, search_result, extracts):
    """
    Build a Markdown string from the Tavily search result and extraction data.
    Returns JSON-formatted string with markdown_report and list_of_images.
    """
    # Track all images found in the report
    all_images = []
    
    if not search_result:
        return json.dumps({
            "markdown_report": f"# Tavily Search Error\nNo valid search results for query: {query}",
            "images": all_images
        })

    # Convert extract response to a URL-keyed dictionary if it's not already
    extracts_by_url = {}
    if isinstance(extracts, list):
        for item in extracts:
            if isinstance(item, dict) and "url" in item:
                extracts_by_url[item["url"]] = item
    elif isinstance(extracts, dict):
        extracts_by_url = extracts

    answer = search_result.get("answer", "No summary provided.")
    items = search_result.get("results", [])
    
    # Extract search metadata
    search_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_period = f"Past {search_result.get('days', 7)} days"

    md_lines = []
    md_lines.append(f"# Crime Report Search Results")
    md_lines.append(f"**Query:** `{query}`")
    md_lines.append(f"**Search Time:** {search_date}")
    md_lines.append(f"**Time Period:** {time_period}\n")
    
    # Add a fallback image at the top if needed
    images_in_report = False
    for item in items:
        if "image_url" in item and item["image_url"]:
            images_in_report = True
            break
            
    if not images_in_report:
        fallback = get_fallback_images()
        if fallback:
            md_lines.append(f"![Crime Scene]({fallback[0]})\n")
            all_images.append(fallback[0])
    
    md_lines.append(f"## Summary / Answer\n{answer}\n")
    md_lines.append("## Detailed Results\n")

    if not items:
        md_lines.append("*No search items found.*")
        return json.dumps({
            "markdown_report": "\n".join(md_lines),
            "images": all_images
        })

    for i, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "#")
        snippet = item.get("content", "No snippet.")
        
        md_lines.append(f"**{i}.** [{title}]({url})")
        
        # Add images if available in search results
        if "image_url" in item and item["image_url"]:
            img_url = item["image_url"]
            if img_url and img_url.startswith("http"):  # Ensure URL is valid
                md_lines.append(f"\n![Image from {title}]({img_url})")
                all_images.append(img_url)
        
        md_lines.append(f"\n> {snippet}\n")

        # If we have extracted text for this URL, add it
        if url in extracts_by_url:
            extracted_data = extracts_by_url[url]
            if isinstance(extracted_data, dict):
                # Add extracted text
                text_content = extracted_data.get("text", "")
                if len(text_content) > 500:
                    text_content = text_content[:500] + "..."
                md_lines.append(f"**Extracted Content**:\n\n{text_content}\n")
                
                # Add extracted images
                if "images" in extracted_data and extracted_data["images"]:
                    md_lines.append("**Images:**\n")
                    for img in extracted_data["images"][:3]:  # Limit to first 3 images
                        if isinstance(img, dict) and "url" in img:
                            img_url = img["url"]
                            if img_url and img_url.startswith("http"):  # Ensure URL is valid
                                img_desc = img.get("description", "Image from article")
                                md_lines.append(f"![{img_desc}]({img_url})\n")
                                all_images.append(img_url)

    return json.dumps({
        "markdown_report": "\n".join(md_lines),
        "images": all_images
    })

# ---------------------------------------------------------------------------
# 5. Main Usage Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    location = "New York City"
    query_str = f"Recent crime incidents and statistics"
    
    # List of news and government domains that typically have reliable crime data
    reliable_domains = [
        "nytimes.com", "cbsnews.com", "nypost.com", "ny1.com", 
        "nyc.gov", "nypd.org", "abc7ny.com", "nbcnewyork.com",
        "apnews.com", "reuters.com", "cnn.com"
    ]

    # 1) Do a Tavily search
    search_response = tavily_search(
        query=query_str,
        topic="news",
        search_depth="advanced",
        max_results=5,
        days=3,
        include_domains=reliable_domains,
        exclude_domains=["twitter.com", "reddit.com", "facebook.com"],
        location=location
    )
    
    if not search_response:
        print("Search failed. Exiting.")
        exit()

    # 2) Extract content from the found URLs
    urls = [item["url"] for item in search_response.get("results", []) if "url" in item]
    extract_response = tavily_extract(urls=urls, extract_depth="basic", include_images=True)
    
    # Process the extract response and add images to search results
    if extract_response and isinstance(extract_response, list):
        # Create a URL-to-images mapping
        for extract_item in extract_response:
            if isinstance(extract_item, dict) and "url" in extract_item and "images" in extract_item:
                url = extract_item["url"]
                # Find the matching search result
                for search_item in search_response.get("results", []):
                    if search_item.get("url") == url:
                        # If this search result has no image_url but extract has images, use the first image
                        if not search_item.get("image_url") and extract_item["images"]:
                            first_image = extract_item["images"][0]
                            if isinstance(first_image, dict) and "url" in first_image:
                                search_item["image_url"] = first_image["url"]
                                print(f"Added image from extract to search result: {first_image['url']}")
                        break
        
        # Create extracts_by_url dictionary as before
        extracts_by_url = {}
        for item in extract_response:
            if isinstance(item, dict) and "url" in item:
                extracts_by_url[item["url"]] = item
    else:
        extracts_by_url = {}

    # 3) Build a neat Markdown report
    # ...existing code...

    result_json = build_markdown_report(query_str, search_response, extracts_by_url)
    result_data = json.loads(result_json)

    # 4) Write to file
    report_file = "crime_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(result_data["markdown_report"])

    print(f"\nReport generated and saved to: {report_file}")
    print(f"Found {len(result_data['images'])} images in the report")
    print("Open the report in a Markdown viewer to see the images properly.")
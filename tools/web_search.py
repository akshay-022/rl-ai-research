import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_response(xml_data):
    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("atom:entry", NS):
        title = entry.find("atom:title", NS)
        summary = entry.find("atom:summary", NS)
        id_elem = entry.find("atom:id", NS)
        published = entry.find("atom:published", NS)

        url = id_elem.text if id_elem is not None else ""
        arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else ""

        authors = []
        for author in entry.findall("atom:author", NS):
            name = author.find("atom:name", NS)
            if name is not None and name.text:
                authors.append(name.text)

        papers.append({
            "title": " ".join(title.text.split()) if title is not None and title.text else "",
            "abstract": " ".join(summary.text.split()) if summary is not None and summary.text else None,
            "url": url,
            "arxiv_id": arxiv_id,
            "year": int(published.text[:4]) if published is not None and published.text else None,
            "authors": authors[:5],
        })

    return papers


def web_search(query, max_results=5):
    if not query or not query.strip():
        return {"results": [], "query": query, "error": "Empty query"}

    try:
        # Filter to papers from last 2 years
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")

        params = urllib.parse.urlencode({
            "search_query": f"all:{query} AND submittedDate:[{cutoff} TO *]",
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": "relevance",
            "sortOrder": "descending",
        })

        req = urllib.request.Request(f"{ARXIV_API}?{params}", headers={"User-Agent": "Python"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            papers = _parse_response(resp.read().decode("utf-8"))

        return {"results": papers, "query": query, "error": None}
    except Exception as e:
        return {"results": [], "query": query, "error": str(e)}


def get_paper_by_id(arxiv_id):
    arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "").strip()
    if not arxiv_id:
        return {"error": "Empty arXiv ID"}

    try:
        params = urllib.parse.urlencode({"id_list": arxiv_id})
        req = urllib.request.Request(f"{ARXIV_API}?{params}", headers={"User-Agent": "Python"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            papers = _parse_response(resp.read().decode("utf-8"))

        if not papers:
            return {"error": f"Paper not found: {arxiv_id}"}

        return {**papers[0], "error": None}
    except Exception as e:
        return {"error": str(e)}


# Tool definitions for task.py
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search for ML/AI papers on arXiv",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5}
        },
        "required": ["query"],
    },
}

GET_PAPER_TOOL = {
    "name": "get_paper_details",
    "description": "Get paper details by arXiv ID",
    "input_schema": {
        "type": "object",
        "properties": {
            "arxiv_id": {"type": "string", "description": "arXiv ID (e.g. 2104.09864)"}
        },
        "required": ["arxiv_id"],
    },
}

TOOLS = [WEB_SEARCH_TOOL, GET_PAPER_TOOL]

HANDLERS = {
    "web_search": lambda query, max_results=5: web_search(query, max_results),
    "get_paper_details": lambda arxiv_id: get_paper_by_id(arxiv_id),
}

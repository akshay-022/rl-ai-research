"""
Tools for literature review task.
Provides paper search, TLDR summaries, and reference retrieval.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

# Semantic Scholar API
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = "gJOElmNEvP6tx12G5RVa05tEsQcAPYT34IbubhEn"


def _parse_arxiv_response(xml_data):
    """Parse arXiv API XML response into paper dicts."""
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


def _search_semantic_scholar(query, max_results=5):
    """Search using Semantic Scholar API."""
    from datetime import datetime, timedelta
    cutoff_year = (datetime.now() - timedelta(days=730)).year

    params = urllib.parse.urlencode({
        "query": query,
        "fields": "paperId,title,abstract,url,year,authors",
        "limit": min(max_results, 100),
        "year": f"{cutoff_year}-",
    })

    headers = {
        "User-Agent": "Python",
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
    }

    req = urllib.request.Request(
        f"{SEMANTIC_SCHOLAR_API}/paper/search?{params}",
        headers=headers
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    papers = []
    for paper in data.get("data", []):
        authors = []
        for author in (paper.get("authors") or [])[:5]:
            if author.get("name"):
                authors.append(author["name"])

        papers.append({
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract"),
            "url": paper.get("url") or f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
            "paper_id": paper.get("paperId"),
            "year": paper.get("year"),
            "authors": authors,
            "source": "semantic_scholar",
        })

    return papers


def web_search(query, max_results=5):
    """
    Search for ML/AI papers. Tries arXiv first, falls back to Semantic Scholar.
    """
    if not query or not query.strip():
        return {"results": [], "query": query, "error": "Empty query"}

    arxiv_error = None

    # Try arXiv first
    try:
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
            papers = _parse_arxiv_response(resp.read().decode("utf-8"))

        for p in papers:
            p["source"] = "arxiv"
            p["paper_id"] = p.get("arxiv_id")

        return {"results": papers, "query": query, "error": None, "source": "arxiv"}
    except Exception as e:
        arxiv_error = str(e)

    # Fallback to Semantic Scholar
    try:
        papers = _search_semantic_scholar(query, max_results)
        return {"results": papers, "query": query, "error": None, "source": "semantic_scholar"}
    except Exception as e:
        return {"results": [], "query": query, "error": f"arXiv: {arxiv_error}; Semantic Scholar: {str(e)}"}


def get_paper_with_tldr(paper_id):
    """
    Get detailed paper info including AI-generated TLDR summary.
    paper_id can be: arXiv ID (e.g., "2104.09864"), Semantic Scholar ID, or DOI.
    """
    if not paper_id or not str(paper_id).strip():
        return {"error": "Empty paper ID"}

    # Normalize arXiv IDs to include prefix
    paper_id = str(paper_id).strip()
    if paper_id.replace(".", "").replace("v", "").isdigit() or (len(paper_id.split(".")) == 2):
        if not paper_id.lower().startswith("arxiv:"):
            paper_id = f"arXiv:{paper_id}"

    try:
        fields = "paperId,title,abstract,tldr,authors,year,citationCount,openAccessPdf,url"
        params = urllib.parse.urlencode({"fields": fields})

        headers = {
            "User-Agent": "Python",
            "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        }

        req = urllib.request.Request(
            f"{SEMANTIC_SCHOLAR_API}/paper/{urllib.parse.quote(paper_id, safe='')}?{params}",
            headers=headers
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        authors = []
        for author in (data.get("authors") or [])[:5]:
            if author.get("name"):
                authors.append(author["name"])

        tldr = None
        if data.get("tldr") and data["tldr"].get("text"):
            tldr = data["tldr"]["text"]

        pdf_url = None
        if data.get("openAccessPdf") and data["openAccessPdf"].get("url"):
            pdf_url = data["openAccessPdf"]["url"]

        return {
            "paper_id": data.get("paperId"),
            "title": data.get("title", ""),
            "abstract": data.get("abstract"),
            "tldr": tldr,
            "authors": authors,
            "year": data.get("year"),
            "citation_count": data.get("citationCount"),
            "pdf_url": pdf_url,
            "url": data.get("url"),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def get_paper_references(paper_id, limit=10):
    """
    Get papers that a given paper cites (its references).
    Useful for finding foundational/related work mentioned in a paper's intro.
    """
    if not paper_id or not str(paper_id).strip():
        return {"error": "Empty paper ID", "references": []}

    paper_id = str(paper_id).strip()
    if paper_id.replace(".", "").replace("v", "").isdigit() or (len(paper_id.split(".")) == 2):
        if not paper_id.lower().startswith("arxiv:"):
            paper_id = f"arXiv:{paper_id}"

    try:
        fields = "paperId,title,abstract,year,authors,citationCount"
        params = urllib.parse.urlencode({
            "fields": fields,
            "limit": min(limit, 100),
        })

        headers = {
            "User-Agent": "Python",
            "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        }

        req = urllib.request.Request(
            f"{SEMANTIC_SCHOLAR_API}/paper/{urllib.parse.quote(paper_id, safe='')}/references?{params}",
            headers=headers
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        references = []
        for ref in data.get("data", []):
            cited_paper = ref.get("citedPaper", {})
            if not cited_paper or not cited_paper.get("title"):
                continue

            authors = []
            for author in (cited_paper.get("authors") or [])[:3]:
                if author.get("name"):
                    authors.append(author["name"])

            references.append({
                "paper_id": cited_paper.get("paperId"),
                "title": cited_paper.get("title"),
                "abstract": cited_paper.get("abstract"),
                "year": cited_paper.get("year"),
                "authors": authors,
                "citation_count": cited_paper.get("citationCount"),
            })

        return {
            "paper_id": paper_id,
            "references": references,
            "count": len(references),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "references": []}


# Tool definitions for Anthropic API
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search for ML/AI papers on arXiv (with Semantic Scholar fallback). Returns titles, abstracts, and paper IDs.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5}
        },
        "required": ["query"],
    },
}

GET_PAPER_TLDR_TOOL = {
    "name": "get_paper_with_tldr",
    "description": "Get detailed paper info including AI-generated TLDR summary. Use this to get a concise summary of a paper's key contributions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Paper ID - can be arXiv ID (e.g. '2104.09864'), Semantic Scholar ID, or DOI"
            }
        },
        "required": ["paper_id"],
    },
}

GET_PAPER_REFS_TOOL = {
    "name": "get_paper_references",
    "description": "Get papers that a given paper cites (its references). Useful for finding foundational/related work.",
    "input_schema": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "Paper ID - can be arXiv ID (e.g. '2104.09864'), Semantic Scholar ID, or DOI"
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of references to return"
            }
        },
        "required": ["paper_id"],
    },
}

TOOLS = [WEB_SEARCH_TOOL, GET_PAPER_TLDR_TOOL, GET_PAPER_REFS_TOOL]

HANDLERS = {
    "web_search": lambda query, max_results=5: web_search(query, max_results),
    "get_paper_with_tldr": lambda paper_id: get_paper_with_tldr(paper_id),
    "get_paper_references": lambda paper_id, limit=10: get_paper_references(paper_id, limit),
}

"""
Tools for literature review task.
Provides paper search, TLDR summaries, and reference retrieval.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

# Semantic Scholar API
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = "gJOElmNEvP6tx12G5RVa05tEsQcAPYT34IbubhEn"

# Rate limiting - track last request time
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 1.0  # Minimum seconds between requests


def _rate_limit():
    """Enforce rate limiting between API requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _request_with_retry(url, headers=None, max_retries=3, timeout=15):
    """Make HTTP request with retry logic for rate limits."""
    _rate_limit()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "Python"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                print(f"  ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif e.code >= 500:  # Server error
                wait_time = (attempt + 1) * 1
                print(f"  ⚠️ Server error {e.code}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise

    raise Exception(f"Failed after {max_retries} retries")


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

    url = f"{SEMANTIC_SCHOLAR_API}/paper/search?{params}"
    response_text = _request_with_retry(url, headers=headers)
    data = json.loads(response_text)

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

        url = f"{ARXIV_API}?{params}"
        response_text = _request_with_retry(url, headers={"User-Agent": "Python"})
        papers = _parse_arxiv_response(response_text)

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

        url = f"{SEMANTIC_SCHOLAR_API}/paper/{urllib.parse.quote(paper_id, safe='')}?{params}"
        response_text = _request_with_retry(url, headers=headers)
        data = json.loads(response_text)

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
    Get papers that a given paper cites (its references) WITH their abstracts.
    The abstracts provide context about what each referenced work contributes.
    """
    if not paper_id or not str(paper_id).strip():
        return {"error": "Empty paper ID", "references": []}

    paper_id = str(paper_id).strip()
    if paper_id.replace(".", "").replace("v", "").isdigit() or (len(paper_id.split(".")) == 2):
        if not paper_id.lower().startswith("arxiv:"):
            paper_id = f"arXiv:{paper_id}"

    try:
        fields = "paperId,title,abstract,year,authors,citationCount,tldr"
        params = urllib.parse.urlencode({
            "fields": fields,
            "limit": min(limit, 100),
        })

        headers = {
            "User-Agent": "Python",
            "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        }

        url = f"{SEMANTIC_SCHOLAR_API}/paper/{urllib.parse.quote(paper_id, safe='')}/references?{params}"
        response_text = _request_with_retry(url, headers=headers)
        data = json.loads(response_text)

        references = []
        for ref in data.get("data", []):
            cited_paper = ref.get("citedPaper", {})
            if not cited_paper or not cited_paper.get("title"):
                continue

            authors = []
            for author in (cited_paper.get("authors") or [])[:3]:
                if author.get("name"):
                    authors.append(author["name"])

            # Get TLDR if available
            tldr = None
            if cited_paper.get("tldr") and cited_paper["tldr"].get("text"):
                tldr = cited_paper["tldr"]["text"]

            references.append({
                "paper_id": cited_paper.get("paperId"),
                "title": cited_paper.get("title"),
                "abstract": cited_paper.get("abstract"),
                "tldr": tldr,
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


def _extract_introduction_from_pdf(pdf_url):
    """
    Download PDF and extract the introduction section.
    The introduction contains the literature review context we need.
    """
    import io
    import re

    try:
        # Try to import PyPDF2 or pdfplumber
        try:
            import PyPDF2
            use_pypdf = True
        except ImportError:
            try:
                import pdfplumber
                use_pypdf = False
            except ImportError:
                return None, "PDF parsing libraries not available (install PyPDF2 or pdfplumber)"

        # Download PDF with retry logic
        _rate_limit()
        max_retries = 3
        pdf_data = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(pdf_url, headers={"User-Agent": "Python/research-agent"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    pdf_data = resp.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    wait_time = (attempt + 1) * 2
                    print(f"  ⏳ PDF download error {e.code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise

        if pdf_data is None:
            return None, "Failed to download PDF after retries"

        # Extract text from first few pages (intro is usually in first 3-4 pages)
        if use_pypdf:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            text = ""
            for i in range(min(5, len(pdf_reader.pages))):
                text += pdf_reader.pages[i].extract_text() + "\n"
        else:
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                text = ""
                for i in range(min(5, len(pdf.pages))):
                    text += pdf.pages[i].extract_text() + "\n"

        # Try to extract just the introduction section
        intro_text = _find_introduction_section(text)
        return intro_text, None

    except Exception as e:
        return None, str(e)


def _find_introduction_section(text):
    """
    Extract the introduction section from paper text.
    Introduction typically contains the literature review.
    """
    import re

    # Common section header patterns
    intro_patterns = [
        r'(?:^|\n)\s*1\.?\s*Introduction\s*\n',
        r'(?:^|\n)\s*I\.?\s*Introduction\s*\n',
        r'(?:^|\n)\s*INTRODUCTION\s*\n',
    ]

    # Patterns that mark end of introduction
    end_patterns = [
        r'(?:^|\n)\s*2\.?\s*(?:Related\s*Work|Background|Preliminaries|Method)',
        r'(?:^|\n)\s*II\.?\s*(?:Related\s*Work|Background|Preliminaries|Method)',
        r'(?:^|\n)\s*(?:RELATED\s*WORK|BACKGROUND|PRELIMINARIES|METHOD)',
    ]

    # Find introduction start
    intro_start = None
    for pattern in intro_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            intro_start = match.end()
            break

    if intro_start is None:
        # No clear intro section, return first ~4000 chars
        return text[:4000]

    # Find introduction end
    intro_end = len(text)
    remaining_text = text[intro_start:]
    for pattern in end_patterns:
        match = re.search(pattern, remaining_text, re.IGNORECASE)
        if match:
            intro_end = intro_start + match.start()
            break

    intro_text = text[intro_start:intro_end].strip()

    # Limit length but try to include full introduction
    if len(intro_text) > 6000:
        intro_text = intro_text[:6000] + "\n[... truncated]"

    return intro_text


def get_paper_introduction(paper_id):
    """
    Get the INTRODUCTION section of a paper - this is where authors discuss
    related work, compare approaches, and provide literature context.
    Much more valuable than abstracts for understanding the research landscape.
    """
    if not paper_id or not str(paper_id).strip():
        return {"error": "Empty paper ID"}

    paper_id = str(paper_id).strip()

    # First, get paper metadata from Semantic Scholar to find PDF URL
    arxiv_id = None
    pdf_url = None

    if paper_id.lower().startswith("arxiv:"):
        arxiv_id = paper_id[6:]
    elif paper_id.replace(".", "").replace("v", "").isdigit():
        arxiv_id = paper_id
    elif len(paper_id.split(".")) == 2 and paper_id.split(".")[0].isdigit():
        arxiv_id = paper_id

    # Try Semantic Scholar for PDF URL
    title = ""
    abstract = ""
    try:
        headers = {
            "User-Agent": "Python",
            "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        }
        url = f"{SEMANTIC_SCHOLAR_API}/paper/{urllib.parse.quote(paper_id, safe='')}?fields=externalIds,title,openAccessPdf,abstract"
        response_text = _request_with_retry(url, headers=headers)
        data = json.loads(response_text)

        title = data.get("title", "")
        abstract = data.get("abstract", "")

        if data.get("openAccessPdf") and data["openAccessPdf"].get("url"):
            pdf_url = data["openAccessPdf"]["url"]

        ext_ids = data.get("externalIds", {})
        if ext_ids.get("ArXiv"):
            arxiv_id = ext_ids["ArXiv"]

    except Exception:
        pass

    # If we have arXiv ID, construct PDF URL
    if arxiv_id and not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    if not pdf_url:
        return {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "introduction": None,
            "error": "No PDF available for this paper",
        }

    # Extract introduction from PDF
    intro_text, error = _extract_introduction_from_pdf(pdf_url)

    if error:
        # Fall back to just abstract
        return {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "introduction": None,
            "pdf_url": pdf_url,
            "error": f"Could not extract introduction: {error}. Use abstract instead.",
        }

    return {
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "introduction": intro_text,
        "pdf_url": pdf_url,
        "error": None,
    }


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
    "description": "Get papers that a given paper cites (its references) WITH abstracts and TLDRs. This provides context about what each referenced work contributes - useful for understanding the literature landscape.",
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

GET_PAPER_INTRO_TOOL = {
    "name": "get_paper_introduction",
    "description": "Extract the INTRODUCTION section from a paper's PDF. The introduction is where authors discuss related work, compare approaches, and provide comprehensive literature context. This is MUCH more valuable than abstracts for understanding the research landscape. Use this on survey papers or seminal works to get their literature review.",
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

TOOLS = [WEB_SEARCH_TOOL, GET_PAPER_TLDR_TOOL, GET_PAPER_REFS_TOOL, GET_PAPER_INTRO_TOOL]

HANDLERS = {
    "web_search": lambda query, max_results=5: web_search(query, max_results),
    "get_paper_with_tldr": lambda paper_id: get_paper_with_tldr(paper_id),
    "get_paper_references": lambda paper_id, limit=10: get_paper_references(paper_id, limit),
    "get_paper_introduction": lambda paper_id: get_paper_introduction(paper_id),
}

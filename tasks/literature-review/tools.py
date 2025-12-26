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
    """Make HTTP request with retry logic for rate limits (429 only)."""
    _rate_limit()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "Python"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited - retry with backoff
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                print(f"  ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                # Don't retry other errors (including 500s)
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


def _download_pdf(pdf_url):
    """Download PDF and return raw bytes."""
    _rate_limit()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "Python/research-agent"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read(), None
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Only retry rate limits
                wait_time = (attempt + 1) * 2
                print(f"  ⏳ PDF rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            return None, f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, str(e)
    return None, "Failed to download PDF after retries"


def _extract_pdf_text(pdf_data, start_page=0, end_page=None):
    """Extract text from PDF pages."""
    import io

    try:
        import PyPDF2
        use_pypdf = True
    except ImportError:
        try:
            import pdfplumber
            use_pypdf = False
        except ImportError:
            return None, "PDF parsing libraries not available (install PyPDF2 or pdfplumber)"

    try:
        if use_pypdf:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            num_pages = len(pdf_reader.pages)
            if end_page is None:
                end_page = num_pages
            end_page = min(end_page, num_pages)

            text = ""
            for i in range(start_page, end_page):
                text += pdf_reader.pages[i].extract_text() + "\n"
        else:
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                num_pages = len(pdf.pages)
                if end_page is None:
                    end_page = num_pages
                end_page = min(end_page, num_pages)

                text = ""
                for i in range(start_page, end_page):
                    text += pdf.pages[i].extract_text() + "\n"

        return text, None
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


def _find_methodology_section(text):
    """
    Extract the methodology/methods section from paper text.
    This explains how the approach works.
    """
    import re

    # Common section header patterns for methodology
    method_patterns = [
        r'(?:^|\n)\s*\d+\.?\s*(?:Method(?:s|ology)?|Approach|Model|Architecture|Framework)\s*\n',
        r'(?:^|\n)\s*(?:II|III|IV)\.?\s*(?:Method(?:s|ology)?|Approach|Model|Architecture)\s*\n',
        r'(?:^|\n)\s*(?:METHOD(?:S|OLOGY)?|APPROACH|MODEL|ARCHITECTURE)\s*\n',
        r'(?:^|\n)\s*\d+\.?\s*(?:Proposed|Our)\s+(?:Method|Approach|Model)\s*\n',
    ]

    # Patterns that mark end of methodology
    end_patterns = [
        r'(?:^|\n)\s*\d+\.?\s*(?:Experiment|Result|Evaluation|Implementation)',
        r'(?:^|\n)\s*(?:IV|V|VI)\.?\s*(?:Experiment|Result|Evaluation)',
        r'(?:^|\n)\s*(?:EXPERIMENT|RESULT|EVALUATION)',
    ]

    # Find methodology start
    method_start = None
    for pattern in method_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            method_start = match.end()
            break

    if method_start is None:
        return None

    # Find methodology end
    method_end = len(text)
    remaining_text = text[method_start:]
    for pattern in end_patterns:
        match = re.search(pattern, remaining_text, re.IGNORECASE)
        if match:
            method_end = method_start + match.start()
            break

    method_text = text[method_start:method_end].strip()

    # Limit length
    if len(method_text) > 6000:
        method_text = method_text[:6000] + "\n[... truncated]"

    return method_text


def _find_results_section(text):
    """
    Extract the results/experiments section from paper text.
    This contains quantitative findings and performance numbers.
    """
    import re

    # Common section header patterns for results
    results_patterns = [
        r'(?:^|\n)\s*\d+\.?\s*(?:Experiment(?:s|al)?|Result(?:s)?|Evaluation|Empirical)\s*\n',
        r'(?:^|\n)\s*(?:IV|V|VI)\.?\s*(?:Experiment|Result|Evaluation)\s*\n',
        r'(?:^|\n)\s*(?:EXPERIMENT|RESULT|EVALUATION)\s*\n',
        r'(?:^|\n)\s*\d+\.?\s*(?:Experiment(?:s|al)?\s+(?:Result|Setup|Evaluation))\s*\n',
    ]

    # Patterns that mark end of results
    end_patterns = [
        r'(?:^|\n)\s*\d+\.?\s*(?:Conclusion|Discussion|Limitation|Related\s*Work|Future)',
        r'(?:^|\n)\s*(?:V|VI|VII)\.?\s*(?:Conclusion|Discussion|Limitation)',
        r'(?:^|\n)\s*(?:CONCLUSION|DISCUSSION|LIMITATION)',
        r'(?:^|\n)\s*(?:Acknowledg|Reference)',
    ]

    # Find results start
    results_start = None
    for pattern in results_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            results_start = match.end()
            break

    if results_start is None:
        return None

    # Find results end
    results_end = len(text)
    remaining_text = text[results_start:]
    for pattern in end_patterns:
        match = re.search(pattern, remaining_text, re.IGNORECASE)
        if match:
            results_end = results_start + match.start()
            break

    results_text = text[results_start:results_end].strip()

    # Limit length
    if len(results_text) > 6000:
        results_text = results_text[:6000] + "\n[... truncated]"

    return results_text


def _get_paper_metadata(paper_id):
    """Get paper metadata and PDF URL from Semantic Scholar."""
    paper_id = str(paper_id).strip()

    arxiv_id = None
    pdf_url = None
    title = ""
    abstract = ""

    # Check if it's an arXiv ID
    if paper_id.lower().startswith("arxiv:"):
        arxiv_id = paper_id[6:]
    elif paper_id.replace(".", "").replace("v", "").isdigit():
        arxiv_id = paper_id
    elif len(paper_id.split(".")) == 2 and paper_id.split(".")[0].isdigit():
        arxiv_id = paper_id

    # Try Semantic Scholar for metadata
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

    return {
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "pdf_url": pdf_url,
    }


def get_paper_introduction(paper_id):
    """
    Get the INTRODUCTION section of a paper - this is where authors discuss
    related work, compare approaches, and provide literature context.
    Much more valuable than abstracts for understanding the research landscape.
    """
    if not paper_id or not str(paper_id).strip():
        return {"error": "Empty paper ID"}

    metadata = _get_paper_metadata(paper_id)

    if not metadata["pdf_url"]:
        return {
            **metadata,
            "introduction": None,
            "error": "No PDF available for this paper",
        }

    # Download PDF
    pdf_data, error = _download_pdf(metadata["pdf_url"])
    if error:
        return {
            **metadata,
            "introduction": None,
            "error": f"Could not download PDF: {error}",
        }

    # Extract text from first pages (intro is usually in first 5 pages)
    text, error = _extract_pdf_text(pdf_data, start_page=0, end_page=5)
    if error:
        return {
            **metadata,
            "introduction": None,
            "error": f"Could not extract text: {error}",
        }

    intro_text = _find_introduction_section(text)

    return {
        **metadata,
        "introduction": intro_text,
        "error": None,
    }


def get_paper_methodology(paper_id):
    """
    Get the METHODOLOGY/METHODS section of a paper - this explains how the
    approach works, the architecture, and technical details.
    """
    if not paper_id or not str(paper_id).strip():
        return {"error": "Empty paper ID"}

    metadata = _get_paper_metadata(paper_id)

    if not metadata["pdf_url"]:
        return {
            **metadata,
            "methodology": None,
            "error": "No PDF available for this paper",
        }

    # Download PDF
    pdf_data, error = _download_pdf(metadata["pdf_url"])
    if error:
        return {
            **metadata,
            "methodology": None,
            "error": f"Could not download PDF: {error}",
        }

    # Extract text from pages 2-10 (methods usually after intro)
    text, error = _extract_pdf_text(pdf_data, start_page=1, end_page=10)
    if error:
        return {
            **metadata,
            "methodology": None,
            "error": f"Could not extract text: {error}",
        }

    method_text = _find_methodology_section(text)

    if not method_text:
        return {
            **metadata,
            "methodology": None,
            "error": "Could not find methodology section in paper",
        }

    return {
        **metadata,
        "methodology": method_text,
        "error": None,
    }


def get_paper_results(paper_id):
    """
    Get the RESULTS/EXPERIMENTS section of a paper - this contains quantitative
    findings, performance numbers, benchmarks, and comparisons.
    """
    if not paper_id or not str(paper_id).strip():
        return {"error": "Empty paper ID"}

    metadata = _get_paper_metadata(paper_id)

    if not metadata["pdf_url"]:
        return {
            **metadata,
            "results": None,
            "error": "No PDF available for this paper",
        }

    # Download PDF
    pdf_data, error = _download_pdf(metadata["pdf_url"])
    if error:
        return {
            **metadata,
            "results": None,
            "error": f"Could not download PDF: {error}",
        }

    # Extract text from pages 5-15 (results usually in middle/end)
    text, error = _extract_pdf_text(pdf_data, start_page=4, end_page=15)
    if error:
        return {
            **metadata,
            "results": None,
            "error": f"Could not extract text: {error}",
        }

    results_text = _find_results_section(text)

    if not results_text:
        return {
            **metadata,
            "results": None,
            "error": "Could not find results section in paper",
        }

    return {
        **metadata,
        "results": results_text,
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
    "description": "Extract the INTRODUCTION section from a paper's PDF. The introduction is where authors discuss related work, compare approaches, and provide comprehensive literature context. Use this to understand how papers relate to each other.",
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

GET_PAPER_METHOD_TOOL = {
    "name": "get_paper_methodology",
    "description": "Extract the METHODOLOGY/METHODS section from a paper's PDF. This explains how the approach works, the architecture, algorithms, and technical details.",
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

GET_PAPER_RESULTS_TOOL = {
    "name": "get_paper_results",
    "description": "Extract the RESULTS/EXPERIMENTS section from a paper's PDF. This contains quantitative findings, performance numbers, benchmarks, and comparisons with other methods.",
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

TOOLS = [WEB_SEARCH_TOOL, GET_PAPER_TLDR_TOOL, GET_PAPER_REFS_TOOL, GET_PAPER_INTRO_TOOL, GET_PAPER_METHOD_TOOL, GET_PAPER_RESULTS_TOOL]

HANDLERS = {
    "web_search": lambda query, max_results=5: web_search(query, max_results),
    "get_paper_with_tldr": lambda paper_id: get_paper_with_tldr(paper_id),
    "get_paper_references": lambda paper_id, limit=10: get_paper_references(paper_id, limit),
    "get_paper_introduction": lambda paper_id: get_paper_introduction(paper_id),
    "get_paper_methodology": lambda paper_id: get_paper_methodology(paper_id),
    "get_paper_results": lambda paper_id: get_paper_results(paper_id),
}

import time
import sys
import os

# Add parent directory to path so we can import from tools/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from web_search import web_search, get_paper_by_id, get_paper_with_tldr, get_paper_references

def test_search():
    print("Testing search...")
    results = web_search("rotary position embedding", max_results=3)
    if results["error"]:
        print(f"FAIL: {results['error']}")
        return False
    print(f"OK: found {len(results['results'])} papers")
    for p in results["results"]:
        print(f"  - {p['title'][:60]}...")
    return True

def test_get_paper():
    print("\nTesting get_paper_by_id...")
    paper = get_paper_by_id("2104.09864")
    if paper.get("error"):
        print(f"FAIL: {paper['error']}")
        return False
    print(f"OK: {paper['title']}")
    return True

def test_get_paper_tldr():
    print("\nTesting get_paper_with_tldr...")
    paper = get_paper_with_tldr("2104.09864")
    if paper.get("error"):
        print(f"FAIL: {paper['error']}")
        return False
    print(f"OK: {paper['title']}")
    print(f"  TLDR: {paper.get('tldr', 'N/A')[:100]}...")
    print(f"  Citations: {paper.get('citation_count', 'N/A')}")
    return True

def test_get_references():
    print("\nTesting get_paper_references...")
    refs = get_paper_references("2104.09864", limit=5)
    if refs.get("error"):
        print(f"FAIL: {refs['error']}")
        return False
    print(f"OK: found {refs['count']} references")
    for r in refs["references"][:3]:
        print(f"  - {r['title'][:60]}...")
    return True

if __name__ == "__main__":
    time.sleep(1)
    t1 = test_search()
    time.sleep(1)
    t2 = test_get_paper()
    time.sleep(1)
    t3 = test_get_paper_tldr()
    time.sleep(1)
    t4 = test_get_references()

    all_passed = t1 and t2 and t3 and t4
    print(f"\n{'All passed' if all_passed else 'Some failed'}")

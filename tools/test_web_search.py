import time
from web_search import web_search, get_paper_by_id

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

if __name__ == "__main__":
    time.sleep(1)
    t1 = test_search()
    time.sleep(1)
    t2 = test_get_paper()
    print(f"\n{'All passed' if t1 and t2 else 'Some failed'}")

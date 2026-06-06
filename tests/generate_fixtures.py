"""
Run once on master branch to produce golden fixture files:
  python -m tests.generate_fixtures
"""
import json
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from WikiWho.wikiwho import Wikiwho

ARTICLES = [
    "Adam Himebauch",
    "Splatoon 3",
]

API_URL = "https://en.wikipedia.org/w/api.php"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


HEADERS = {"User-Agent": "WikiWho-regression-tests/1.0 (https://github.com/wikimedia/WikiWho; xenacode-art)"}


def fetch_revisions(title):
    import time
    revisions = []
    params = {
        "action": "query", "format": "json", "titles": title,
        "prop": "revisions",
        "rvprop": "content|ids|timestamp|sha1|comment|flags|user|userid",
        "rvlimit": "max", "rvdir": "newer", "continue": "",
    }
    while True:
        for attempt in range(5):
            try:
                resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=60).json()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"  Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        page = next(iter(resp["query"]["pages"].values()))
        revisions.extend(page.get("revisions", []))
        if "continue" not in resp:
            break
        params.update(resp["continue"])
    return revisions


def slug(title):
    return title.lower().replace(" ", "_")


for title in ARTICLES:
    rev_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_revisions.json")
    golden_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_golden.json")

    if not os.path.exists(rev_path):
        print(f"Fetching revisions for '{title}'...")
        revisions = fetch_revisions(title)
        with open(rev_path, "w", encoding="utf-8") as f:
            json.dump(revisions, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(revisions)} revisions.")
    else:
        with open(rev_path, encoding="utf-8") as f:
            revisions = json.load(f)
        print(f"Using cached revisions for '{title}' ({len(revisions)} revs)")

    print(f"Running WikiWho on '{title}'...")
    ww = Wikiwho(title)
    ww.analyse_article(revisions)

    tokens = [
        {
            "str": t.value,
            "token_id": t.token_id,
            "o_rev_id": t.origin_rev_id,
            "in": list(t.inbound),
            "out": list(t.outbound),
        }
        for t in ww.tokens
    ]
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(tokens)} tokens to {golden_path}\n")

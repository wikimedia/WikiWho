"""
Generate cached revision fixtures, and optionally golden token fixtures.

Examples:
  python -m tests.generate_fixtures "Adam Himebauch" "Splatoon 3"
  python -m tests.generate_fixtures "Japan Cup" --through-revid 524409649 --revisions-only
"""
import argparse
import json
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from WikiWho.wikiwho import Wikiwho

API_URL = "https://en.wikipedia.org/w/api.php"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


HEADERS = {"User-Agent": "WikiWho-regression-tests/1.0 (https://github.com/wikimedia/WikiWho; xenacode-art)"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate WikiWho test fixtures.")
    parser.add_argument("titles", nargs="+", help="Wikipedia article title(s) to fetch")
    parser.add_argument(
        "--revisions-only",
        action="store_true",
        help="only write *_revisions.json and skip golden token output",
    )
    parser.add_argument(
        "--through-revid",
        type=int,
        help="stop after this revision id is fetched",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="overwrite an existing *_revisions.json fixture",
    )
    parser.add_argument(
        "--force-golden",
        action="store_true",
        help="overwrite an existing *_golden.json fixture",
    )
    return parser.parse_args()


def fetch_revisions(title, through_revid=None):
    import time
    revisions = []
    params = {
        "action": "query", "format": "json", "titles": title,
        "prop": "revisions",
        "rvprop": "content|ids|timestamp|sha1|comment|flags|user|userid",
        "rvlimit": "max", "rvdir": "newer", "continue": "",
        "redirects": "1",
    }
    while True:
        for attempt in range(5):
            try:
                response = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
                if response.status_code == 429 and response.headers.get("Retry-After"):
                    time.sleep(int(response.headers["Retry-After"]))
                    continue
                response.raise_for_status()
                resp = response.json()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"  Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        page = next(iter(resp["query"]["pages"].values()))
        page_revisions = page.get("revisions", [])
        if through_revid is not None:
            for index, revision in enumerate(page_revisions):
                if int(revision["revid"]) == through_revid:
                    revisions.extend(page_revisions[:index + 1])
                    return revisions
        revisions.extend(page_revisions)
        if "continue" not in resp:
            break
        params.update(resp["continue"])
    if through_revid is not None:
        raise RuntimeError(f"Revision {through_revid} was not found in '{title}' history")
    return revisions


def slug(title):
    return title.lower().replace(" ", "_")


def load_or_fetch_revisions(title, rev_path, args):
    if os.path.exists(rev_path) and not args.force_fetch:
        with open(rev_path, encoding="utf-8") as f:
            revisions = json.load(f)
        print(f"Using cached revisions for '{title}' ({len(revisions)} revs)")
        return revisions

    print(f"Fetching revisions for '{title}'...")
    revisions = fetch_revisions(title, through_revid=args.through_revid)
    with open(rev_path, "w", encoding="utf-8") as f:
        json.dump(revisions, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(revisions)} revisions.")
    return revisions


def write_golden_fixture(title, golden_path, revisions, force=False):
    if os.path.exists(golden_path) and not force:
        print(f"Skipping existing golden fixture {golden_path}; use --force-golden to overwrite.")
        return

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
    print(f"  Saved {len(tokens)} tokens to {golden_path}")


def main():
    args = parse_args()
    for title in args.titles:
        rev_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_revisions.json")
        golden_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_golden.json")

        revisions = load_or_fetch_revisions(title, rev_path, args)
        if not args.revisions_only:
            write_golden_fixture(title, golden_path, revisions, force=args.force_golden)
        print()


if __name__ == "__main__":
    main()

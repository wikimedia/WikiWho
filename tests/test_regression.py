import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from WikiWho.wikiwho import Wikiwho

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

ARTICLES = [
    "Adam Himebauch",
    "Splatoon 3",
]


def slug(title):
    return title.lower().replace(" ", "_")


def load_fixture(title):
    rev_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_revisions.json")
    golden_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_golden.json")
    if not os.path.exists(rev_path) or not os.path.exists(golden_path):
        pytest.skip(
            f"Fixtures missing for '{title}' — run 'python -m tests.generate_fixtures' on master first"
        )
    with open(rev_path, encoding="utf-8") as f:
        revisions = json.load(f)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    return revisions, golden


@pytest.mark.parametrize("title", ARTICLES)
def test_token_authorship_matches_master(title):
    revisions, golden = load_fixture(title)

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

    assert len(tokens) == len(golden), (
        f"Token count mismatch: got {len(tokens)}, expected {len(golden)}"
    )
    for i, (got, want) in enumerate(zip(tokens, golden)):
        assert got == want, (
            f"token[{i}] ({got['str']!r}) mismatch:\n  got:  {got}\n  want: {want}"
        )

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from WikiWho.utils import iter_rev_tokens, split_into_tokens
from WikiWho.wikiwho import Wikiwho

TESTS_DIR = os.path.dirname(__file__)
CASES_PATH = os.path.join(TESTS_DIR, "authorship_cases.json")
ANALYSIS_CACHE = {}


def load_case_groups():
    with open(CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def flatten_cases():
    flattened = []
    for group in load_case_groups():
        for case in group["cases"]:
            flattened.append(pytest.param(group, case, id="{}::{}".format(group["title"], case["id"])))
    return flattened


def fixture_path(group):
    path = group["fixture"]
    if os.path.isabs(path):
        return path
    return os.path.join(TESTS_DIR, path)


def load_revisions(path, history_through_revid=None):
    if not os.path.exists(path):
        pytest.skip("Fixture missing: {}".format(path))
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    revisions = payload["revisions"] if isinstance(payload, dict) else payload
    if history_through_revid is None:
        return revisions

    truncated = []
    for revision in revisions:
        truncated.append(revision)
        if revision["revid"] == history_through_revid:
            return truncated
    raise AssertionError("{} does not contain revision {}".format(path, history_through_revid))


def analysed_article(group):
    path = fixture_path(group)
    through_revid = group.get("history_through_revid")
    cache_key = (group["title"], os.path.abspath(path), through_revid)
    if cache_key not in ANALYSIS_CACHE:
        revisions = load_revisions(path, through_revid)
        wikiwho = Wikiwho(group["title"])
        wikiwho.analyse_article(revisions)
        ANALYSIS_CACHE[cache_key] = wikiwho
    return ANALYSIS_CACHE[cache_key]


def token_values(text):
    return split_into_tokens(text.lower())


def find_subsequence(values, needle, occurrence=1):
    found = 0
    limit = len(values) - len(needle) + 1
    for index in range(max(0, limit)):
        if values[index:index + len(needle)] == needle:
            found += 1
            if found == occurrence:
                return index
    return None


def focus_indices(case, context_tokens):
    indices = []
    for raw_focus in case["focus"]:
        if isinstance(raw_focus, dict):
            text = raw_focus["text"]
            occurrence = raw_focus.get("occurrence", 1)
        else:
            text = raw_focus
            occurrence = 1
        focus_tokens = token_values(text)
        start = find_subsequence(context_tokens, focus_tokens, occurrence=occurrence)
        if start is None:
            raise AssertionError("Focus {!r} not found in {}".format(text, case["id"]))
        indices.extend(range(start, start + len(focus_tokens)))
    return sorted(set(indices))


def serialize_word(index, word):
    return {
        "index": index,
        "value": word.value,
        "token_id": word.token_id,
        "origin_rev_id": word.origin_rev_id,
        "last_rev_id": word.last_rev_id,
        "inbound": list(word.inbound),
        "outbound": list(word.outbound),
    }


def selected_words(group, case):
    wikiwho = analysed_article(group)
    snapshot_revid = case.get("snapshot_revid", group.get("snapshot_revid"))
    if snapshot_revid is None:
        snapshot_revid = wikiwho.ordered_revisions[-1]

    words = list(iter_rev_tokens(wikiwho.revisions[snapshot_revid]))
    if "token_index" in case:
        index = case["token_index"]
        return [serialize_word(index, words[index])]

    values = [word.value for word in words]
    context_tokens = token_values(case["context"])
    context_start = find_subsequence(values, context_tokens, occurrence=case.get("context_occurrence", 1))
    if context_start is None:
        raise AssertionError("Context not found for {}".format(case["id"]))

    selected = []
    for relative_index in focus_indices(case, context_tokens):
        absolute_index = context_start + relative_index
        selected.append(serialize_word(absolute_index, words[absolute_index]))
    return selected


def expected_values(expected, key, selected):
    values = expected[key]
    if isinstance(values, list):
        return values
    return [values] * len(selected)


def assert_expected(selected, expected):
    field_map = {
        "values": "value",
        "token_ids": "token_id",
        "origin_rev_ids": "origin_rev_id",
        "last_rev_ids": "last_rev_id",
        "inbound": "inbound",
        "outbound": "outbound",
    }
    for expected_key, actual_key in field_map.items():
        if expected_key not in expected:
            continue
        actual = [word[actual_key] for word in selected]
        assert actual == expected_values(expected, expected_key, selected)


@pytest.mark.parametrize(("group", "case"), flatten_cases())
def test_authorship_case(group, case):
    selected = selected_words(group, case)
    assert_expected(selected, case["expected"])

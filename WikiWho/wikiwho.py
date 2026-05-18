# -*- coding: utf-8 -*-
"""

:Authors:
    Maribel Acosta,
    Fabian Floeck,
    Andriy Rodchenko,
    Kenan Erdogan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from bisect import bisect_left
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .structures import Word, Sentence, Paragraph, Revision
from .utils import calculate_hash, split_into_paragraphs, split_into_sentences, split_into_tokens, \
    compute_avg_word_freq, iter_rev_tokens


# Spam detection variables.
CHANGE_PERCENTAGE = -0.40
PREVIOUS_LENGTH = 1000
CURR_LENGTH = 1000
FLAG = "move"
UNMATCHED_PARAGRAPH = 0.0
TOKEN_DENSITY_LIMIT = 20
TOKEN_LEN = 100

# Caps estimated identical-token prev/current pairs for the SequenceMatcher opcode pass. Above this, the whole unmatched middle span falls back to bounded nearest-neighbor matching.
WORD_MATCH_MAX_SEQUENCE_PAIRS = 200000

# Caps nearest-neighbor recovery inside one SequenceMatcher replace opcode region. Higher values preserve more matches in broad edits but can reintroduce expensive local scans.
WORD_MATCH_MAX_LOCAL_PAIRS = 10000

# Minimum positional drift allowed for nearest-neighbor reuse of a previous Word object.
WORD_MATCH_MAX_DRIFT_MIN = 50

# Ratio-based nearest-neighbor drift allowed, computed against the larger unmatched side. Higher drift preserves more heuristic matches; lower drift bounds cost and cross-section matches.
WORD_MATCH_MAX_DRIFT_RATIO = 0.10

# Match confidence is a precedence ladder for competing claims to a current token. Cheap local reuse is weakest, exact edge matches are strongest, and structural fixes sit below moved-run recovery so a verified moved copy can still win.
WORD_MATCH_CONF_LOCAL = 20
WORD_MATCH_CONF_SEQUENCE_EQUAL = 90
WORD_MATCH_CONF_STRUCTURAL_BOUNDARY = 92
WORD_MATCH_CONF_MOVED_RUN = 95
WORD_MATCH_CONF_EDGE = 100

# Moved-run recovery looks for unique informative n-grams in unmatched diff regions. The sizes/caps below bound how much extra indexing we do per word diff while still finding copied or moved runs that SequenceMatcher misses.
WORD_MATCH_MOVE_NGRAM_SIZES = (10, 8, 6, 4, 3)
WORD_MATCH_MOVE_MIN_INFO_TOKENS = 3
WORD_MATCH_MOVE_TOKEN_WINDOW = 4
WORD_MATCH_MOVE_MIN_RECOVERABLE_TOKENS = 24
WORD_MATCH_MOVE_MAX_WINDOWS = 300000

# A large pure deletion immediately before an unchanged suffix can otherwise keep very old glue words alive by edge matching. Limit that correction to the first suffix window so normal suffix preservation remains cheap.
WORD_MATCH_EDGE_STALE_REWRITE_MIN_TOKENS = 24
WORD_MATCH_EDGE_STALE_WINDOW = 64

# Structural punctuation is allowed to ride along with a verified moved run, but it cannot seed one by itself.
WORD_MATCH_MOVE_STRUCTURAL_TOKENS = frozenset((
    '.', ',', ';', ':', '?', '!', '-', '_', '/', '\\', '(', ')', '[', ']', '{', '}', '*', '#', '@',
    '&', '=', '+', '%', '~', '$', '^', '<', '>', '"', "'", '|', '{{', '}}', '[[', ']]',
))

# Wikitext constructs whose boundary tokens are too generic to claim through a cheap common-prefix/common-suffix match. If an edge match stops inside one of these constructs, the edge is rolled back so contextual matching sees the whole template/link/comment.
WIKITEXT_CONSTRUCT_PAIRS = (
    ('{{', '}}'),
    ('[[', ']]'),
    ('<!--', '-->'),
)
WIKITEXT_OPEN_TO_CLOSE = dict(WIKITEXT_CONSTRUCT_PAIRS)
WIKITEXT_CLOSE_TO_OPEN = dict((close, open_) for open_, close in WIKITEXT_CONSTRUCT_PAIRS)


def _common_prefix_len(left, right):
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _common_suffix_len(left, right, prefix_len):
    limit = min(len(left), len(right)) - prefix_len
    suffix_len = 0
    while suffix_len < limit and left[len(left) - suffix_len - 1] == right[len(right) - suffix_len - 1]:
        suffix_len += 1
    return suffix_len


def _construct_stack_at(tokens, end):
    stack = []
    for index in range(end):
        token = tokens[index]
        if token in WIKITEXT_OPEN_TO_CLOSE:
            stack.append((token, index))
        elif token in WIKITEXT_CLOSE_TO_OPEN:
            open_token = WIKITEXT_CLOSE_TO_OPEN[token]
            for stack_index in range(len(stack) - 1, -1, -1):
                if stack[stack_index][0] == open_token:
                    del stack[stack_index:]
                    break
    return stack


def _construct_end_after_boundary(tokens, start, open_token):
    close_token = WIKITEXT_OPEN_TO_CLOSE[open_token]
    depth = 1
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token == open_token:
            depth += 1
        elif token == close_token:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _rollback_prefix_construct_boundary(tokens, prefix_len):
    rollback = prefix_len
    while rollback:
        stack = _construct_stack_at(tokens, rollback)
        if not stack:
            return rollback
        open_token, open_index = stack[-1]
        if _construct_end_after_boundary(tokens, rollback, open_token) is None:
            return rollback
        rollback = open_index
    return rollback


def _suffix_construct_boundary_drop(tokens, suffix_start):
    drop = 0
    while suffix_start + drop < len(tokens):
        boundary = suffix_start + drop
        stack = _construct_stack_at(tokens, boundary)
        if not stack:
            return drop
        open_token, _ = stack[-1]
        construct_end = _construct_end_after_boundary(tokens, boundary, open_token)
        if construct_end is None:
            return drop
        drop = construct_end - suffix_start
    return drop


def _rollback_common_construct_edges(left, right, left_keys, right_keys, prefix_len):
    # Do not let cheap edge matches claim generic tokens from inside templates, links, or comments before the contextual matcher sees the whole construct.
    prefix_len = min(_rollback_prefix_construct_boundary(left, prefix_len),
                     _rollback_prefix_construct_boundary(right, prefix_len))
    suffix_len = _common_suffix_len(left_keys, right_keys, prefix_len)
    if suffix_len:
        left_drop = _suffix_construct_boundary_drop(left, len(left) - suffix_len)
        right_drop = _suffix_construct_boundary_drop(right, len(right) - suffix_len)
        suffix_len -= min(suffix_len, max(left_drop, right_drop))
    return prefix_len, suffix_len


def _tokens_until(tokens, start, stops):
    collected = []
    index = start
    while index < len(tokens) and tokens[index] not in stops:
        collected.append(tokens[index])
        index += 1
    return tuple(collected), index


def _template_name_after(tokens, start):
    name, _ = _tokens_until(tokens, start, ('|', '}}'))
    return name


def _link_target_after(tokens, start):
    target, _ = _tokens_until(tokens, start, ('|', ']]'))
    return target


def _template_field_after(tokens, start):
    field, index = _tokens_until(tokens, start, ('=', '|', '}}'))
    if index < len(tokens) and tokens[index] == '=' and field:
        return field
    return None


def _template_field_before(tokens, equals_index):
    field = []
    index = equals_index - 1
    while index >= 0 and tokens[index] not in ('{{', '|', '}}'):
        field.append(tokens[index])
        index -= 1
    if index >= 0 and tokens[index] == '|' and field:
        field.reverse()
        return tuple(field)
    return None


def _link_option_after(tokens, start):
    option, _ = _tokens_until(tokens, start, ('|', ']]'))
    return option


def _pop_construct(stack, construct_type):
    for stack_index in range(len(stack) - 1, -1, -1):
        if stack[stack_index]['type'] == construct_type:
            frame = stack[stack_index]
            del stack[stack_index:]
            return frame
    return None


# Normal tokens still match by value. Low-information wikitext tokens match by local syntax context so, for example, a link option "|" does not match an infobox field "|" and a "{{cite web}}" opener does not match "{{for-multi}}".
def _word_match_keys(tokens):
    keys = list(tokens)
    stack = []
    for index, token in enumerate(tokens):
        if token == '{{':
            name = _template_name_after(tokens, index + 1)
            keys[index] = ('wikitext', '{{', 'template', name) if name else token
            stack.append({'type': 'template', 'name': name, 'arg_index': 0})
        elif token == '}}':
            frame = _pop_construct(stack, 'template')
            name = frame['name'] if frame else None
            keys[index] = ('wikitext', '}}', 'template', name) if name else token
        elif token == '[[':
            target = _link_target_after(tokens, index + 1)
            keys[index] = ('wikitext', '[[', 'link', target) if target else token
            stack.append({'type': 'link', 'target': target, 'option_index': 0})
        elif token == ']]':
            frame = _pop_construct(stack, 'link')
            target = frame['target'] if frame else None
            keys[index] = ('wikitext', ']]', 'link', target) if target else token
        elif token == '<!--':
            keys[index] = ('wikitext', '<!--', 'comment')
            stack.append({'type': 'comment'})
        elif token == '-->':
            _pop_construct(stack, 'comment')
            keys[index] = ('wikitext', '-->', 'comment')
        elif token == '|' and stack:
            frame = stack[-1]
            if frame['type'] == 'link':
                option = _link_option_after(tokens, index + 1)
                keys[index] = ('wikitext', '|', 'link', frame['target'],
                               frame['option_index'], option)
                frame['option_index'] += 1
            elif frame['type'] == 'template':
                field = _template_field_after(tokens, index + 1)
                if field:
                    keys[index] = ('wikitext', '|', 'template-field',
                                   frame['name'], field)
                else:
                    keys[index] = ('wikitext', '|', 'template-arg',
                                   frame['name'], frame['arg_index'])
                frame['arg_index'] += 1
        elif token == '=' and stack and stack[-1]['type'] == 'template':
            field = _template_field_before(tokens, index)
            if field:
                keys[index] = ('wikitext', '=', 'template-field',
                               stack[-1]['name'], field)
    return keys


def _link_spans(tokens):
    # Capture only the target portion of each internal link. Boundary recovery is intentionally disabled for piped links because target/display edits need stricter handling than plain target extension.
    spans = []
    stack = []
    for index, token in enumerate(tokens):
        if token == '[[':
            target, target_end = _tokens_until(tokens, index + 1, ('|', ']]'))
            stack.append({
                'open': index,
                'target_start': index + 1,
                'target_end': target_end,
                'target': target,
                'has_option': target_end < len(tokens) and tokens[target_end] == '|',
            })
        elif token == ']]' and stack:
            frame = stack.pop()
            frame['close'] = index
            spans.append(frame)
    return spans


def _find_contiguous_subsequence(tokens, needle):
    if not needle or len(needle) > len(tokens):
        return None
    limit = len(tokens) - len(needle) + 1
    for start in range(limit):
        if tokens[start:start + len(needle)] == needle:
            return start
    return None


def _link_target_reused(prev_link, curr_link, prev_for_curr):
    if prev_link['has_option'] or curr_link['has_option']:
        return False

    prev_target = prev_link['target']
    curr_target = curr_link['target']
    if not prev_target or not curr_target:
        return False

    target_offset = _find_contiguous_subsequence(curr_target, prev_target)
    if target_offset is None:
        return False
    # Require the old target to remain a substantial part of the new target. This handles target extensions without letting short shared prefixes make unrelated links inherit each other's boundary tokens.
    if len(prev_target) * 2 < len(curr_target):
        return False

    for offset in range(len(prev_target)):
        curr_index = curr_link['target_start'] + target_offset + offset
        prev_index = prev_link['target_start'] + offset
        if curr_index >= len(prev_for_curr) or prev_for_curr[curr_index] != prev_index:
            return False
    return True


def _recover_edited_link_boundaries(text_prev, text_curr, prev_for_curr, match_conf, prev_used_by):
    # Link delimiters are keyed by full target in _word_match_keys, so an edited target can leave [[ and ]] unmatched even when the target body was reused. Recover those delimiters after the body tokens have already matched.
    curr_links_by_first = defaultdict(list)
    for curr_link in _link_spans(text_curr):
        if curr_link['target']:
            curr_links_by_first[curr_link['target'][0]].append(curr_link)

    for prev_link in _link_spans(text_prev):
        if not prev_link['target']:
            continue
        for curr_link in curr_links_by_first.get(prev_link['target'][0], ()):
            if not _link_target_reused(prev_link, curr_link, prev_for_curr):
                continue
            _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                               curr_link['open'], prev_link['open'],
                               WORD_MATCH_CONF_STRUCTURAL_BOUNDARY)
            _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                               curr_link['close'], prev_link['close'],
                               WORD_MATCH_CONF_STRUCTURAL_BOUNDARY)
            break


def _word_match_drift_limit(prev_len, curr_len):
    return max(WORD_MATCH_MAX_DRIFT_MIN,
               int(WORD_MATCH_MAX_DRIFT_RATIO * max(prev_len, curr_len)))


def _word_match_pair_estimate(prev_tokens, curr_tokens):
    prev_counts = Counter(prev_tokens)
    curr_counts = Counter(curr_tokens)
    total = 0
    for token, prev_count in prev_counts.items():
        total += prev_count * curr_counts.get(token, 0)
    return total


def _nearest_word_matches(prev_tokens, curr_tokens, prev_offset, curr_offset, max_drift):
    positions_by_token = defaultdict(list)
    for prev_index, token in enumerate(prev_tokens):
        positions_by_token[token].append(prev_index)

    curr_to_prev = {}
    used_prev = set()
    for curr_index, token in enumerate(curr_tokens):
        positions = positions_by_token.get(token)
        if not positions:
            continue

        expected_prev_index = curr_offset + curr_index - prev_offset
        right = bisect_left(positions, expected_prev_index)
        left = right - 1
        best_prev = None
        best_distance = None
        curr_abs_index = curr_offset + curr_index

        while left >= 0 or right < len(positions):
            if left >= 0:
                prev_index = positions[left]
                distance = abs((prev_offset + prev_index) - curr_abs_index)
                if distance > max_drift or (best_distance is not None and distance > best_distance):
                    left = -1
                else:
                    if prev_index not in used_prev and (
                            best_distance is None or distance < best_distance or
                            (distance == best_distance and prev_index < best_prev)):
                        best_prev = prev_index
                        best_distance = distance
                    left -= 1
            if right < len(positions):
                prev_index = positions[right]
                distance = abs((prev_offset + prev_index) - curr_abs_index)
                if distance > max_drift or (best_distance is not None and distance > best_distance):
                    right = len(positions)
                else:
                    if prev_index not in used_prev and (
                            best_distance is None or distance < best_distance or
                            (distance == best_distance and prev_index < best_prev)):
                        best_prev = prev_index
                        best_distance = distance
                    right += 1

        if best_prev is not None:
            curr_to_prev[curr_index] = best_prev
            used_prev.add(best_prev)
    return curr_to_prev


def _assign_word_match(prev_for_curr, match_conf, prev_used_by, curr_index, prev_index, confidence):
    old_prev = prev_for_curr[curr_index]
    if old_prev is not None:
        if match_conf[curr_index] >= confidence:
            return False
        if prev_used_by.get(old_prev) == curr_index:
            del prev_used_by[old_prev]

    old_curr = prev_used_by.get(prev_index)
    if old_curr is not None:
        if match_conf[old_curr] >= confidence:
            return False
        prev_for_curr[old_curr] = None
        match_conf[old_curr] = 0

    prev_for_curr[curr_index] = prev_index
    match_conf[curr_index] = confidence
    prev_used_by[prev_index] = curr_index
    return True


def _unassign_word_match(prev_for_curr, match_conf, prev_used_by, curr_index):
    prev_index = prev_for_curr[curr_index]
    if prev_index is None:
        return False
    if prev_used_by.get(prev_index) == curr_index:
        del prev_used_by[prev_index]
    prev_for_curr[curr_index] = None
    match_conf[curr_index] = 0
    return True


def _is_low_authorship_edge_token(token):
    return isinstance(token, str) and len(token) <= 2 and token.isalpha()


def _demote_stale_suffix_edge_matches(text_prev, text_curr, prev_words,
                                      prev_for_curr, match_conf, prev_used_by,
                                      prefix_len, suffix_len):
    # Edge suffixes are usually reliable, but after a large pure deletion they can over-preserve old "glue" tokens at the start of a mature rewritten suffix. Demote only those low-authorship tokens and leave content words and replacement edits alone.
    if not prev_words or not suffix_len:
        return

    prev_rewritten_tokens = len(text_prev) - prefix_len - suffix_len
    curr_rewritten_tokens = len(text_curr) - prefix_len - suffix_len
    if curr_rewritten_tokens != 0:
        return
    if prev_rewritten_tokens < WORD_MATCH_EDGE_STALE_REWRITE_MIN_TOKENS:
        return

    suffix_curr_start = len(text_curr) - suffix_len
    limit = min(suffix_len, WORD_MATCH_EDGE_STALE_WINDOW)
    for offset in range(limit):
        curr_index = suffix_curr_start + offset
        prev_index = prev_for_curr[curr_index]
        if prev_index is None or match_conf[curr_index] != WORD_MATCH_CONF_EDGE:
            continue
        if prev_index >= len(prev_words):
            continue
        word_prev = prev_words[prev_index]
        if word_prev.origin_rev_id == word_prev.last_rev_id:
            continue
        if _is_low_authorship_edge_token(text_curr[curr_index]):
            _unassign_word_match(prev_for_curr, match_conf, prev_used_by, curr_index)


def _contiguous_spans(indices):
    spans = []
    if not indices:
        return spans
    start = indices[0]
    previous = start
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
        else:
            spans.append((start, previous + 1))
            start = index
            previous = index
    spans.append((start, previous + 1))
    return spans


def _is_informative_move_token(token):
    return isinstance(token, str) and token not in WORD_MATCH_MOVE_STRUCTURAL_TOKENS and any(
        char.isalnum() for char in token
    )


def _informative_move_token_prefix(tokens):
    prefix = [0]
    total = 0
    for token in tokens:
        if _is_informative_move_token(token):
            total += 1
        prefix.append(total)
    return prefix


def _index_move_ngrams(keys, spans, ngram_size, informative_prefix):
    index = defaultdict(list)
    for start, end in spans:
        for position in range(start, end - ngram_size + 1):
            info_count = informative_prefix[position + ngram_size] - informative_prefix[position]
            if info_count >= WORD_MATCH_MOVE_MIN_INFO_TOKENS:
                key = tuple(keys[position:position + ngram_size])
                index[key].append(position)
    return index


def _content_runs(tokens):
    run = []
    for token in tokens:
        if _is_informative_move_token(token):
            run.append(token)
        elif run:
            yield run
            run = []
    if run:
        yield run


def _longest_content_core(tokens):
    runs = list(_content_runs(tokens))
    if not runs:
        return ()
    return tuple(max(runs, key=len))


def _count_subsequence(tokens, needle):
    if not needle:
        return 0
    count = 0
    needle_len = len(needle)
    for index in range(len(tokens) - needle_len + 1):
        if tuple(tokens[index:index + needle_len]) == needle:
            count += 1
    return count


def _tokens_equal_at(tokens, start, needle):
    for offset, token in enumerate(needle):
        if tokens[start + offset] != token:
            return False
    return True


def _pair_positions(tokens, count_state):
    # Moved-run safety only needs to distinguish unique from repeated runs. Indexing pair positions lets longer subsequence counts probe a small candidate list instead of scanning the whole article text each time.
    pair_indexes = count_state.setdefault('pair_indexes', {})
    index_key = (id(tokens), len(tokens), 'pair_positions')
    positions = pair_indexes.get(index_key)
    if positions is None:
        positions = defaultdict(list)
        for index in range(len(tokens) - 1):
            positions[(tokens[index], tokens[index + 1])].append(index)
        pair_indexes[index_key] = positions
    return positions


def _count_subsequence_cached(tokens, needle, count_state):
    if not needle:
        return 0
    needle = tuple(needle)
    needle_len = len(needle)
    if count_state is None:
        return _count_subsequence(tokens, needle)

    counts = count_state['counts']
    count_key = (id(tokens), len(tokens), needle)
    count = counts.get(count_key)
    if count is not None:
        return count

    if needle_len == 1:
        count = 0
        for token in tokens:
            if token == needle[0]:
                count += 1
                if count > 1:
                    break
        counts[count_key] = count
        return count

    positions = _pair_positions(tokens, count_state)
    first_positions = positions.get((needle[0], needle[1]), ())
    last_positions = positions.get((needle[-2], needle[-1]), ())
    max_start = len(tokens) - needle_len
    count = 0
    if len(last_positions) < len(first_positions):
        for pair_index in last_positions:
            start = pair_index - needle_len + 2
            if start >= 0 and start <= max_start and _tokens_equal_at(tokens, start, needle):
                count += 1
                if count > 1:
                    break
    else:
        for start in first_positions:
            if start <= max_start and _tokens_equal_at(tokens, start, needle):
                count += 1
                if count > 1:
                    break
    counts[count_key] = count
    return count


def _copy_safe_moved_run(count_text_prev, count_text_curr, text_curr, curr_start, length, count_state):
    core = _longest_content_core(text_curr[curr_start:curr_start + length])
    if len(core) < WORD_MATCH_MOVE_MIN_INFO_TOKENS:
        return False
    return (_count_subsequence_cached(count_text_prev, core, count_state) == 1 and
            _count_subsequence_cached(count_text_curr, core, count_state) == 1)


def _content_run_bounds(tokens, index):
    start = index
    while start > 0 and _is_informative_move_token(tokens[start - 1]):
        start -= 1
    end = index + 1
    while end < len(tokens) and _is_informative_move_token(tokens[end]):
        end += 1
    return start, end


def _has_unique_content_window(count_text_prev, count_text_curr, text_curr, curr_index, count_state):
    if not _is_informative_move_token(text_curr[curr_index]):
        return True

    run_start, run_end = _content_run_bounds(text_curr, curr_index)
    if run_end - run_start < WORD_MATCH_MOVE_TOKEN_WINDOW:
        if run_end - run_start >= WORD_MATCH_MOVE_MIN_INFO_TOKENS:
            needle = tuple(text_curr[run_start:run_end])
            if any(any(char.isdigit() for char in token) for token in needle):
                return (_count_subsequence_cached(count_text_prev, needle, count_state) == 1 and
                        _count_subsequence_cached(count_text_curr, needle, count_state) == 1)
        return False

    earliest = max(run_start, curr_index - WORD_MATCH_MOVE_TOKEN_WINDOW + 1)
    latest = min(curr_index, run_end - WORD_MATCH_MOVE_TOKEN_WINDOW)
    for start in range(earliest, latest + 1):
        needle = tuple(text_curr[start:start + WORD_MATCH_MOVE_TOKEN_WINDOW])
        if (_count_subsequence_cached(count_text_prev, needle, count_state) == 1 and
                _count_subsequence_cached(count_text_curr, needle, count_state) == 1):
            return True
    return False


def _can_assign_moved_match(match_conf, prev_used_by, curr_index, prev_index):
    if match_conf[curr_index] >= WORD_MATCH_CONF_MOVED_RUN:
        return False
    old_curr = prev_used_by.get(prev_index)
    return old_curr is None or match_conf[old_curr] < WORD_MATCH_CONF_MOVED_RUN


def _moved_run_confidence(length, seed_length):
    bonus = min(WORD_MATCH_CONF_EDGE - WORD_MATCH_CONF_MOVED_RUN - 1,
                max(0, length - seed_length))
    return WORD_MATCH_CONF_MOVED_RUN + bonus


def _extend_moved_run(prev_keys, curr_keys, prev_for_curr, match_conf, prev_used_by,
                      prev_start, curr_start, length):
    left = 0
    while curr_start - left - 1 >= 0 and prev_start - left - 1 >= 0:
        curr_index = curr_start - left - 1
        prev_index = prev_start - left - 1
        if prev_keys[prev_index] != curr_keys[curr_index]:
            break
        if not _can_assign_moved_match(match_conf, prev_used_by, curr_index, prev_index):
            break
        left += 1

    right = 0
    while curr_start + length + right < len(curr_keys) and prev_start + length + right < len(prev_keys):
        curr_index = curr_start + length + right
        prev_index = prev_start + length + right
        if prev_keys[prev_index] != curr_keys[curr_index]:
            break
        if not _can_assign_moved_match(match_conf, prev_used_by, curr_index, prev_index):
            break
        right += 1

    return prev_start - left, curr_start - left, length + left + right


def _move_ngram_sizes(recoverable_count):
    sizes = list(WORD_MATCH_MOVE_NGRAM_SIZES)
    while len(sizes) > 2 and recoverable_count * len(sizes) > WORD_MATCH_MOVE_MAX_WINDOWS:
        sizes.pop()
    return sizes


def _recoverable_indices_from_spans(spans, start_allowed):
    indices = []
    for span_start, span_end in spans:
        for index in range(span_start, span_end):
            if start_allowed(index):
                indices.append(index)
    return indices


def _recover_moved_word_runs(text_prev, text_curr, prev_keys, curr_keys,
                             prev_for_curr, match_conf, prev_used_by,
                             full_text_prev=None, full_text_curr=None,
                             get_full_texts=None, prev_candidate_spans=None,
                             curr_candidate_spans=None):
    count_texts = [full_text_prev, full_text_curr]
    if prev_candidate_spans is not None and curr_candidate_spans is not None:
        if not prev_candidate_spans or not curr_candidate_spans:
            return

    recoverable_count = len(text_prev) + len(text_curr)
    if recoverable_count < WORD_MATCH_MOVE_MIN_RECOVERABLE_TOKENS:
        return

    def ensure_count_texts():
        if count_texts[0] is None or count_texts[1] is None:
            if get_full_texts is not None:
                count_texts[0], count_texts[1] = get_full_texts()
            else:
                count_texts[0] = text_prev
                count_texts[1] = text_curr
        return count_texts[0], count_texts[1]

    prev_info_prefix = _informative_move_token_prefix(prev_keys)
    curr_info_prefix = _informative_move_token_prefix(curr_keys)
    count_state = {'counts': {}}
    checked_runs = set()
    for ngram_size in _move_ngram_sizes(recoverable_count):
        protected_prev = set(
            prev_index for curr_index, prev_index in enumerate(prev_for_curr)
            if prev_index is not None and match_conf[curr_index] >= WORD_MATCH_CONF_MOVED_RUN
        )
        if prev_candidate_spans is None:
            recoverable_prev = [
                index for index in range(len(text_prev))
                if index not in protected_prev
            ]
        else:
            recoverable_prev = _recoverable_indices_from_spans(
                prev_candidate_spans,
                lambda index: index not in protected_prev,
            )
        if curr_candidate_spans is None:
            recoverable_curr = [
                index for index, confidence in enumerate(match_conf)
                if confidence < WORD_MATCH_CONF_MOVED_RUN
            ]
        else:
            recoverable_curr = _recoverable_indices_from_spans(
                curr_candidate_spans,
                lambda index: match_conf[index] < WORD_MATCH_CONF_MOVED_RUN,
            )

        prev_spans = _contiguous_spans(recoverable_prev)
        curr_spans = _contiguous_spans(recoverable_curr)
        prev_ngrams = _index_move_ngrams(prev_keys, prev_spans, ngram_size, prev_info_prefix)
        curr_ngrams = _index_move_ngrams(curr_keys, curr_spans, ngram_size, curr_info_prefix)

        candidates = []
        for key, prev_positions in prev_ngrams.items():
            curr_positions = curr_ngrams.get(key)
            if curr_positions and len(prev_positions) == 1 and len(curr_positions) == 1:
                candidates.append((abs(prev_positions[0] - curr_positions[0]),
                                   prev_positions[0], curr_positions[0]))

        for _, prev_start, curr_start in sorted(candidates, reverse=True):
            if any(
                not _can_assign_moved_match(match_conf, prev_used_by,
                                            curr_start + offset, prev_start + offset)
                for offset in range(ngram_size)
            ):
                continue

            prev_start, curr_start, length = _extend_moved_run(
                prev_keys, curr_keys, prev_for_curr, match_conf, prev_used_by,
                prev_start, curr_start, ngram_size,
            )
            run_key = (prev_start, curr_start, length)
            if run_key in checked_runs:
                continue
            checked_runs.add(run_key)
            count_text_prev, count_text_curr = ensure_count_texts()
            if not _copy_safe_moved_run(count_text_prev, count_text_curr, text_curr,
                                        curr_start, length, count_state):
                continue

            confidence = _moved_run_confidence(length, ngram_size)
            for offset in range(length):
                curr_index = curr_start + offset
                if not _has_unique_content_window(count_text_prev, count_text_curr,
                                                  text_curr, curr_index, count_state):
                    continue
                _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                                   curr_index, prev_start + offset,
                                   confidence)


def _match_word_sequences(text_prev, text_curr, full_text_prev=None, full_text_curr=None,
                          get_full_texts=None, prev_words=None):
    prev_for_curr = [None] * len(text_curr)
    match_conf = [0] * len(text_curr)
    prev_used_by = {}

    prev_keys = _word_match_keys(text_prev)
    curr_keys = _word_match_keys(text_curr)

    prefix_len = _common_prefix_len(prev_keys, curr_keys)
    prefix_len, suffix_len = _rollback_common_construct_edges(text_prev, text_curr,
                                                              prev_keys, curr_keys,
                                                              prefix_len)
    for index in range(prefix_len):
        _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                           index, index, WORD_MATCH_CONF_EDGE)
    for index in range(suffix_len):
        prev_index = len(text_prev) - suffix_len + index
        curr_index = len(text_curr) - suffix_len + index
        _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                           curr_index, prev_index, WORD_MATCH_CONF_EDGE)

    prev_mid_start = prefix_len
    prev_mid_end = len(text_prev) - suffix_len
    curr_mid_start = prefix_len
    curr_mid_end = len(text_curr) - suffix_len
    prev_mid = text_prev[prev_mid_start:prev_mid_end]
    curr_mid = text_curr[curr_mid_start:curr_mid_end]
    prev_mid_keys = prev_keys[prev_mid_start:prev_mid_end]
    curr_mid_keys = curr_keys[curr_mid_start:curr_mid_end]
    move_prev_spans = []
    move_curr_spans = []

    if prev_mid and curr_mid:
        max_drift = _word_match_drift_limit(len(prev_mid), len(curr_mid))
        if _word_match_pair_estimate(prev_mid_keys, curr_mid_keys) <= WORD_MATCH_MAX_SEQUENCE_PAIRS:
            matcher = SequenceMatcher(None, prev_mid_keys, curr_mid_keys, autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    for prev_index, curr_index in zip(range(i1, i2), range(j1, j2)):
                        _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                                           curr_mid_start + curr_index,
                                           prev_mid_start + prev_index,
                                           WORD_MATCH_CONF_SEQUENCE_EQUAL)
                else:
                    if tag in ('replace', 'delete') and i1 < i2:
                        move_prev_spans.append((prev_mid_start + i1, prev_mid_start + i2))
                    if tag in ('replace', 'insert') and j1 < j2:
                        move_curr_spans.append((curr_mid_start + j1, curr_mid_start + j2))
                    if tag == 'replace' and (i2 - i1) * (j2 - j1) <= WORD_MATCH_MAX_LOCAL_PAIRS:
                        local_matches = _nearest_word_matches(prev_mid_keys[i1:i2],
                                                              curr_mid_keys[j1:j2],
                                                              prev_mid_start + i1,
                                                              curr_mid_start + j1,
                                                              max_drift)
                        for curr_index, prev_index in local_matches.items():
                            _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                                               curr_mid_start + j1 + curr_index,
                                               prev_mid_start + i1 + prev_index,
                                               WORD_MATCH_CONF_LOCAL)
        else:
            move_prev_spans.append((prev_mid_start, prev_mid_end))
            move_curr_spans.append((curr_mid_start, curr_mid_end))
            local_matches = _nearest_word_matches(prev_mid_keys, curr_mid_keys,
                                                  prev_mid_start, curr_mid_start,
                                                  max_drift)
            for curr_index, prev_index in local_matches.items():
                _assign_word_match(prev_for_curr, match_conf, prev_used_by,
                                   curr_mid_start + curr_index,
                                   prev_mid_start + prev_index,
                                   WORD_MATCH_CONF_LOCAL)

    _recover_moved_word_runs(text_prev, text_curr, prev_keys, curr_keys,
                             prev_for_curr, match_conf, prev_used_by,
                             full_text_prev=full_text_prev,
                             full_text_curr=full_text_curr,
                             get_full_texts=get_full_texts,
                             prev_candidate_spans=move_prev_spans,
                             curr_candidate_spans=move_curr_spans)
    _recover_edited_link_boundaries(text_prev, text_curr, prev_for_curr,
                                    match_conf, prev_used_by)
    _demote_stale_suffix_edge_matches(text_prev, text_curr, prev_words,
                                      prev_for_curr, match_conf, prev_used_by,
                                      prefix_len, suffix_len)

    matched_prev = set(prev_index for prev_index in prev_for_curr if prev_index is not None)
    deleted_prev = [index for index in range(len(text_prev)) if index not in matched_prev]
    return prev_for_curr, deleted_prev


class Wikiwho:
    def __init__(self, article_title):
        # Hash tables.
        self.paragraphs_ht = {}
        self.sentences_ht = {}

        self.spam_ids = []
        # Mutable public list kept for compatibility. Changes to spam_hashes
        # must go through _add_spam_revision() exclusively so spam_hashes_set
        # stays in sync for membership checks.
        self.spam_hashes = []
        self.tokens = []  # [word_obj, ..] ordered, unique list of tokens of this article
        self.revisions = {}  # {rev_id : rev_obj, ...}
        self.ordered_revisions = []  # [rev_id, ...]
        self.rvcontinue = '0'
        self.title = article_title
        self.page_id = None  # article id
        self.token_id = 0  # sequential id for tokens in article. unique per token per article.
        # Revisions to compare.
        self.revision_curr = Revision()
        self.revision_prev = Revision()

        self.text_curr = ''
        self.temp = []

        # Keep the public list shape while using a set for membership checks.
        self.spam_hashes_set = set()

    def _add_spam_revision(self, rev_id, rev_hash):
        """Record a spam revision while keeping the public list and set synced."""
        self.spam_ids.append(rev_id)
        self.spam_hashes.append(rev_hash)
        self.spam_hashes_set.add(rev_hash)

    def clean_attributes(self):
        """
        Empty attributes that are usually not needed after analyzing an article.
        """
        self.revision_prev = None
        self.text_curr = ''
        self.temp = []

    def analyse_article_from_xml_dump(self, page):
        """
        Analyse page from XML Dump Iterator.
        :param page: Page meta data and a Revision iterator. Each revision contains metadata and text.
        """
        # Iterate over revisions of the article.
        for revision in page:
            text = revision.text or ''
            if not text and (revision.deleted.text or revision.deleted.restricted):
                # equivalent of "'texthidden' in revision or 'textmissing' in revision" in analyse_article
                continue

            vandalism = False
            # Update the information about the previous revision.
            self.revision_prev = self.revision_curr

            rev_id = revision.id
            rev_hash = revision.sha1 or calculate_hash(text)
            if rev_hash in self.spam_hashes_set:
                vandalism = True

            # TODO: spam detection: DELETION
            text_len = len(text)
            if not vandalism and not(revision.comment and revision.minor):
                # if content is not moved (flag) to different article in good faith, check for vandalism
                # if revisions have reached a certain size
                if self.revision_prev.length > PREVIOUS_LENGTH and \
                   text_len < CURR_LENGTH and \
                   ((text_len-self.revision_prev.length) / self.revision_prev.length) <= CHANGE_PERCENTAGE:
                    # VANDALISM: CHANGE PERCENTAGE - DELETION
                    vandalism = True

            if vandalism:
                # print("---------------------------- FLAG 1")
                self.revision_curr = self.revision_prev
                self._add_spam_revision(rev_id, rev_hash)
            else:
                # Information about the current revision.
                self.revision_curr = Revision()
                self.revision_curr.id = rev_id
                self.revision_curr.length = text_len
                self.revision_curr.timestamp = revision.timestamp.long_format()

                # Get editor information
                if revision.user:
                    user_text = revision.user.text
                    contributor_name = '' if not user_text or user_text == 'None' else user_text
                    if revision.user.id is None and contributor_name or revision.user.id == 0:
                        contributor_id = 0
                    else:
                        contributor_id = revision.user.id or ''
                else:
                    # Some revisions don't have contributor.
                    contributor_name = ''
                    contributor_id = ''
                editor = contributor_id
                editor = str(editor) if editor != 0 else '0|{}'.format(contributor_name)
                self.revision_curr.editor = editor

                # Content within the revision.
                self.text_curr = text.lower()

                # Perform comparison.
                vandalism = self.determine_authorship()

                if vandalism:
                    # print "---------------------------- FLAG 2"
                    self.revision_curr = self.revision_prev  # skip revision with vandalism in history
                    self._add_spam_revision(rev_id, rev_hash)
                else:
                    # Add the current revision with all the information.
                    self.revisions.update({self.revision_curr.id: self.revision_curr})
                    self.ordered_revisions.append(self.revision_curr.id)
            self.temp = []

    def analyse_article(self, page):
        """
        Analyse page in json form.
        :param page: List of revisions. Each revision is a dict and contains metadata and text.
        """
        # Iterate over revisions of the article.
        for revision in page:
            if 'texthidden' in revision or 'textmissing' in revision:
                continue

            vandalism = False
            # Update the information about the previous revision.
            self.revision_prev = self.revision_curr

            text = revision.get('*', '')
            rev_id = int(revision['revid'])
            rev_hash = revision.get('sha1')
            if not rev_hash:
                rev_hash = calculate_hash(text)
            if rev_hash in self.spam_hashes_set:
                vandalism = True

            # TODO: spam detection: DELETION
            text_len = len(text)
            if not vandalism and not(revision.get('comment') and 'minor' in revision):
                # if content is not moved (flag) to different article in good faith, check for vandalism
                # if revisions have reached a certain size
                if self.revision_prev.length > PREVIOUS_LENGTH and \
                   text_len < CURR_LENGTH and \
                   ((text_len-self.revision_prev.length) / self.revision_prev.length) <= CHANGE_PERCENTAGE:
                    # VANDALISM: CHANGE PERCENTAGE - DELETION
                    vandalism = True

            if vandalism:
                # print("---------------------------- FLAG 1")
                self.revision_curr = self.revision_prev
                self._add_spam_revision(rev_id, rev_hash)
            else:
                # Information about the current revision.
                self.revision_curr = Revision()
                self.revision_curr.id = rev_id
                self.revision_curr.length = text_len
                self.revision_curr.timestamp = revision['timestamp']

                # Get editor information.
                # Some revisions don't have editor.
                contributor_id = revision.get('userid', '')
                contributor_name = revision.get('user', '')
                editor = contributor_id
                editor = str(editor) if editor != 0 else '0|{}'.format(contributor_name)
                self.revision_curr.editor = editor

                # Content within the revision.
                self.text_curr = text.lower()

                # Perform comparison.
                vandalism = self.determine_authorship()

                if vandalism:
                    # print "---------------------------- FLAG 2"
                    self.revision_curr = self.revision_prev  # skip revision with vandalism in history
                    self._add_spam_revision(rev_id, rev_hash)
                else:
                    # Add the current revision with all the information.
                    self.revisions.update({self.revision_curr.id: self.revision_curr})
                    self.ordered_revisions.append(self.revision_curr.id)
            self.temp = []

    def determine_authorship(self):
        # Containers for unmatched paragraphs and sentences in both revisions.
        unmatched_sentences_curr = []
        unmatched_sentences_prev = []
        matched_paragraphs_prev = []
        matched_sentences_prev = []
        matched_words_prev = []
        possible_vandalism = False
        vandalism = False

        try:
            # Analysis of the paragraphs in the current revision.
            unmatched_paragraphs_curr, unmatched_paragraphs_prev, matched_paragraphs_prev = \
                self.analyse_paragraphs_in_revision()

            # Analysis of the sentences in the unmatched paragraphs of the current revision.
            if unmatched_paragraphs_curr:
                unmatched_sentences_curr, unmatched_sentences_prev, matched_sentences_prev, total_sentences = \
                    self.analyse_sentences_in_paragraphs(unmatched_paragraphs_curr, unmatched_paragraphs_prev)

                # TODO: spam detection
                if len(unmatched_paragraphs_curr) / len(self.revision_curr.ordered_paragraphs) > UNMATCHED_PARAGRAPH:
                    # will be used to detect copy-paste vandalism - token density
                    possible_vandalism = True

                # Analysis of words in unmatched sentences (diff of both texts).
                if unmatched_sentences_curr:
                    matched_words_prev, vandalism = self.analyse_words_in_sentences(unmatched_sentences_curr,
                                                                                    unmatched_sentences_prev,
                                                                                    possible_vandalism)
        except Exception:
            # Error occurred during analysing the current revision
            # Hold the last successfully processed revision.
            self.revision_curr = self.revision_prev
            # Reset matched structures from old revisions.
            for matched_paragraph in matched_paragraphs_prev:
                matched_paragraph.matched = False
                for sentence_hash in matched_paragraph.sentences:
                    for sentence in matched_paragraph.sentences[sentence_hash]:
                        sentence.matched = False
                        for word_prev in sentence.words:
                            word_prev.matched = False
            for matched_sentence in matched_sentences_prev:
                matched_sentence.matched = False
                for word_prev in matched_sentence.words:
                    word_prev.matched = False
            for matched_word in matched_words_prev:
                matched_word.matched = False
            raise

        if not vandalism:
            # Add the information of 'deletion' to words
            for unmatched_sentence in unmatched_sentences_prev:
                for word_prev in unmatched_sentence.words:
                    if not word_prev.matched:
                        word_prev.outbound.append(self.revision_curr.id)
            if not unmatched_sentences_prev:
                # if all current paragraphs are matched
                for unmatched_paragraph in unmatched_paragraphs_prev:
                    for sentence_hash in unmatched_paragraph.sentences:
                        for sentence in unmatched_paragraph.sentences[sentence_hash]:
                            for word_prev in sentence.words:
                                if not word_prev.matched:
                                    word_prev.outbound.append(self.revision_curr.id)

        # Reset matched structures from old revisions. And update inbound and last used info of matched words.
        for matched_paragraph in matched_paragraphs_prev:
            matched_paragraph.matched = False
            for sentence_hash in matched_paragraph.sentences:
                for sentence in matched_paragraph.sentences[sentence_hash]:
                    sentence.matched = False
                    for word_prev in sentence.words:
                        # first update inbound and last used info of matched words of all previous revisions
                        if not vandalism and word_prev.matched and \
                                (not word_prev.outbound or word_prev.outbound[-1] != self.revision_curr.id):
                            if word_prev.last_rev_id != self.revision_prev.id:
                                word_prev.inbound.append(self.revision_curr.id)
                            word_prev.last_rev_id = self.revision_curr.id
                        # reset
                        word_prev.matched = False
        for matched_sentence in matched_sentences_prev:
            matched_sentence.matched = False
            for word_prev in matched_sentence.words:
                # first update inbound and last used info of matched words of all previous revisions
                if not vandalism and word_prev.matched and \
                        (not word_prev.outbound or word_prev.outbound[-1] != self.revision_curr.id):
                    if word_prev.last_rev_id != self.revision_prev.id:
                        word_prev.inbound.append(self.revision_curr.id)
                    word_prev.last_rev_id = self.revision_curr.id
                # reset
                word_prev.matched = False
        for matched_word in matched_words_prev:
            # first update last used info of matched prev words
            # there is no inbound chance because we only diff with words of previous revision
            if not vandalism and word_prev.matched:
                if not word_prev.outbound or word_prev.outbound[-1] != self.revision_curr.id:
                    word_prev.last_rev_id = self.revision_curr.id
            # reset
            matched_word.matched = False

        if not vandalism:
            # Add the new paragraphs to hash table of paragraphs.
            for unmatched_paragraph in unmatched_paragraphs_curr:
                if unmatched_paragraph.hash_value in self.paragraphs_ht:
                    self.paragraphs_ht[unmatched_paragraph.hash_value].append(unmatched_paragraph)
                else:
                    self.paragraphs_ht.update({unmatched_paragraph.hash_value: [unmatched_paragraph]})
                unmatched_paragraph.value = ''  # hash value is not used for next rev analysis

            # Add the new sentences to hash table of sentences.
            for unmatched_sentence in unmatched_sentences_curr:
                if unmatched_sentence.hash_value in self.sentences_ht:
                    self.sentences_ht[unmatched_sentence.hash_value].append(unmatched_sentence)
                else:
                    self.sentences_ht.update({unmatched_sentence.hash_value: [unmatched_sentence]})
                unmatched_sentence.value = ''  # hash value is not used for next rev analysis
                unmatched_sentence.splitted = None  # splitted word values are not used for next rev analysis

        return vandalism

    def analyse_paragraphs_in_revision(self):
        # Containers for unmatched and matched paragraphs.
        unmatched_paragraphs_curr = []
        unmatched_paragraphs_prev = []
        matched_paragraphs_prev = []

        # Split the text of the current into paragraphs.
        paragraphs = split_into_paragraphs(self.text_curr)

        # Iterate over the paragraphs of the current version.
        for paragraph in paragraphs:
            # Build Paragraph structure and calculate hash value.
            paragraph = paragraph.strip()
            if not paragraph:
                # dont track empty lines
                continue
            # TODO should we clean whitespaces in paragraph level?
            # paragraph = ' '.join(split_into_tokens(paragraph))
            hash_curr = calculate_hash(paragraph)
            matched_curr = False

            # If the paragraph is in the previous revision,
            # update the authorship information and mark both paragraphs as matched (also in HT).
            for paragraph_prev in self.revision_prev.paragraphs.get(hash_curr, []):
                if not paragraph_prev.matched:
                    matched_one = False
                    matched_all = True
                    for h in paragraph_prev.sentences:
                        for s_prev in paragraph_prev.sentences[h]:
                            for w_prev in s_prev.words:
                                if w_prev.matched:
                                    matched_one = True
                                else:
                                    matched_all = False

                    if not matched_one:
                        # if there is not any already matched prev word, so set them all as matched
                        matched_curr = True
                        paragraph_prev.matched = True
                        matched_paragraphs_prev.append(paragraph_prev)

                        # Set all sentences and words of this paragraph as matched
                        for hash_sentence_prev in paragraph_prev.sentences:
                            for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                                sentence_prev.matched = True
                                for word_prev in sentence_prev.words:
                                    word_prev.matched = True

                        # Add paragraph to current revision.
                        if hash_curr in self.revision_curr.paragraphs:
                            self.revision_curr.paragraphs[hash_curr].append(paragraph_prev)
                        else:
                            self.revision_curr.paragraphs.update({paragraph_prev.hash_value: [paragraph_prev]})
                        self.revision_curr.ordered_paragraphs.append(paragraph_prev.hash_value)
                        break
                    elif matched_all:
                        # if all prev words in this paragraph are already matched
                        paragraph_prev.matched = True
                        # for hash_sentence_prev in paragraph_prev.sentences:
                        #     for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                        #         sentence_prev.matched = True
                        matched_paragraphs_prev.append(paragraph_prev)

            # If the paragraph is not in the previous revision, but it is in an older revision
            # update the authorship information and mark both paragraphs as matched.
            if not matched_curr:
                for paragraph_prev in self.paragraphs_ht.get(hash_curr, []):
                    if not paragraph_prev.matched:
                        matched_one = False
                        matched_all = True
                        for h in paragraph_prev.sentences:
                            for s_prev in paragraph_prev.sentences[h]:
                                for w_prev in s_prev.words:
                                    if w_prev.matched:
                                        matched_one = True
                                    else:
                                        matched_all = False

                        if not matched_one:
                            # if there is not any already matched prev word, so set them all as matched
                            matched_curr = True
                            paragraph_prev.matched = True
                            matched_paragraphs_prev.append(paragraph_prev)

                            # Set all sentences and words of this paragraph as matched
                            for hash_sentence_prev in paragraph_prev.sentences:
                                for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                                    sentence_prev.matched = True
                                    for word_prev in sentence_prev.words:
                                        word_prev.matched = True

                            # Add paragraph to current revision.
                            if hash_curr in self.revision_curr.paragraphs:
                                self.revision_curr.paragraphs[hash_curr].append(paragraph_prev)
                            else:
                                self.revision_curr.paragraphs.update({paragraph_prev.hash_value: [paragraph_prev]})
                            self.revision_curr.ordered_paragraphs.append(paragraph_prev.hash_value)
                            break
                        elif matched_all:
                            # if all prev words in this paragraph are already matched
                            paragraph_prev.matched = True
                            # for hash_sentence_prev in paragraph_prev.sentences:
                            #     for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                            #         sentence_prev.matched = True
                            matched_paragraphs_prev.append(paragraph_prev)

            # If the paragraph did not match with previous revisions,
            # add to container of unmatched paragraphs for further analysis.
            if not matched_curr:
                paragraph_curr = Paragraph()
                paragraph_curr.hash_value = hash_curr
                paragraph_curr.value = paragraph

                if hash_curr in self.revision_curr.paragraphs:
                    self.revision_curr.paragraphs[hash_curr].append(paragraph_curr)
                else:
                    self.revision_curr.paragraphs.update({paragraph_curr.hash_value: [paragraph_curr]})
                self.revision_curr.ordered_paragraphs.append(paragraph_curr.hash_value)
                unmatched_paragraphs_curr.append(paragraph_curr)

        # Identify unmatched paragraphs in previous revision for further analysis.
        paragraph_duplicate_counts = {}
        for paragraph_prev_hash in self.revision_prev.ordered_paragraphs:
            if len(self.revision_prev.paragraphs[paragraph_prev_hash]) > 1:
                count = paragraph_duplicate_counts.get(paragraph_prev_hash, 0) + 1
                paragraph_duplicate_counts[paragraph_prev_hash] = count
                paragraph_prev = self.revision_prev.paragraphs[paragraph_prev_hash][count - 1]
            else:
                paragraph_prev = self.revision_prev.paragraphs[paragraph_prev_hash][0]
            if not paragraph_prev.matched:
                unmatched_paragraphs_prev.append(paragraph_prev)

        return unmatched_paragraphs_curr, unmatched_paragraphs_prev, matched_paragraphs_prev

    def analyse_sentences_in_paragraphs(self, unmatched_paragraphs_curr, unmatched_paragraphs_prev):
        # Containers for unmatched and matched sentences.
        unmatched_sentences_curr = []
        unmatched_sentences_prev = []
        matched_sentences_prev = []
        total_sentences = 0

        # Iterate over the unmatched paragraphs of the current revision.
        for paragraph_curr in unmatched_paragraphs_curr:
            # Split the current paragraph into sentences.
            sentences = split_into_sentences(paragraph_curr.value)
            # Iterate over the sentences of the current paragraph
            for sentence in sentences:
                # Create the Sentence structure.
                sentence = sentence.strip()
                if not sentence:
                    # dont track empty lines
                    continue
                sentence = ' '.join(split_into_tokens(sentence))  # here whitespaces in the sentence are cleaned
                hash_curr = calculate_hash(sentence)  # then hash values is calculated
                matched_curr = False
                total_sentences += 1

                # Iterate over the unmatched paragraphs from the previous revision.
                for paragraph_prev in unmatched_paragraphs_prev:
                    for sentence_prev in paragraph_prev.sentences.get(hash_curr, []):
                        if not sentence_prev.matched:
                            matched_one = False
                            matched_all = True
                            for word_prev in sentence_prev.words:
                                if word_prev.matched:
                                    matched_one = True
                                else:
                                    matched_all = False

                            if not matched_one:
                                # if there is not any already matched prev word, so set them all as matched
                                sentence_prev.matched = True
                                matched_curr = True
                                matched_sentences_prev.append(sentence_prev)

                                for word_prev in sentence_prev.words:
                                    word_prev.matched = True

                                # Add the sentence information to the paragraph.
                                if hash_curr in paragraph_curr.sentences:
                                    paragraph_curr.sentences[hash_curr].append(sentence_prev)
                                else:
                                    paragraph_curr.sentences.update({sentence_prev.hash_value: [sentence_prev]})
                                paragraph_curr.ordered_sentences.append(sentence_prev.hash_value)
                                break
                            elif matched_all:
                                # if all prev words in this sentence are already matched
                                sentence_prev.matched = True
                                matched_sentences_prev.append(sentence_prev)
                    if matched_curr:
                        break

                # Iterate over the hash table of sentences from old revisions.
                if not matched_curr:
                    for sentence_prev in self.sentences_ht.get(hash_curr, []):
                        if not sentence_prev.matched:
                            matched_one = False
                            matched_all = True
                            for word_prev in sentence_prev.words:
                                if word_prev.matched:
                                    matched_one = True
                                else:
                                    matched_all = False

                            if not matched_one:
                                # if there is not any already matched prev word, so set them all as matched
                                sentence_prev.matched = True
                                matched_curr = True
                                matched_sentences_prev.append(sentence_prev)

                                for word_prev in sentence_prev.words:
                                    word_prev.matched = True

                                # Add the sentence information to the paragraph.
                                if hash_curr in paragraph_curr.sentences:
                                    paragraph_curr.sentences[hash_curr].append(sentence_prev)
                                else:
                                    paragraph_curr.sentences.update({sentence_prev.hash_value: [sentence_prev]})
                                paragraph_curr.ordered_sentences.append(sentence_prev.hash_value)
                                break
                            elif matched_all:
                                # if all prev words in this sentence are already matched
                                sentence_prev.matched = True
                                matched_sentences_prev.append(sentence_prev)

                # If the sentence did not match,
                # then include in the container of unmatched sentences for further analysis.
                if not matched_curr:
                    sentence_curr = Sentence()
                    sentence_curr.value = sentence
                    sentence_curr.hash_value = hash_curr

                    if hash_curr in paragraph_curr.sentences:
                        paragraph_curr.sentences[hash_curr].append(sentence_curr)
                    else:
                        paragraph_curr.sentences.update({sentence_curr.hash_value: [sentence_curr]})
                    paragraph_curr.ordered_sentences.append(sentence_curr.hash_value)
                    unmatched_sentences_curr.append(sentence_curr)

        # Identify the unmatched sentences in the previous paragraph revision.
        sentence_duplicate_counts = {}
        for paragraph_prev in unmatched_paragraphs_prev:
            for sentence_prev_hash in paragraph_prev.ordered_sentences:
                if len(paragraph_prev.sentences[sentence_prev_hash]) > 1:
                    key = (id(paragraph_prev), sentence_prev_hash)
                    count = sentence_duplicate_counts.get(key, 0) + 1
                    sentence_duplicate_counts[key] = count
                    sentence_prev = paragraph_prev.sentences[sentence_prev_hash][count - 1]
                else:
                    sentence_prev = paragraph_prev.sentences[sentence_prev_hash][0]
                if not sentence_prev.matched:
                    unmatched_sentences_prev.append(sentence_prev)
                    # to reset 'matched words in analyse_words_in_sentences' of unmatched paragraphs and sentences
                    sentence_prev.matched = True
                    matched_sentences_prev.append(sentence_prev)

        return unmatched_sentences_curr, unmatched_sentences_prev, matched_sentences_prev, total_sentences

    def analyse_words_in_sentences(self, unmatched_sentences_curr, unmatched_sentences_prev, possible_vandalism):
        matched_words_prev = []
        unmatched_words_prev = []

        # Split sentences into words.
        text_prev = []
        for sentence_prev in unmatched_sentences_prev:
            for word_prev in sentence_prev.words:
                if not word_prev.matched:
                    text_prev.append(word_prev.value)
                    unmatched_words_prev.append(word_prev)

        # Build flat (sentence, token) slots so we can assign words during the
        # diff pass without re-scanning sentences or the diff list.
        curr_slots = []  # list of (sentence_curr, word_value)
        text_curr = []
        for sentence_curr in unmatched_sentences_curr:
            # split_into_tokens is already done in analyse_sentences_in_paragraphs
            words = sentence_curr.value.split(' ')
            text_curr.extend(words)
            sentence_curr.splitted.extend(words)
            for word in words:
                curr_slots.append((sentence_curr, word))

        # Edit consists of removing sentences, not adding new content.
        if not text_curr:
            return matched_words_prev, False

        # spam detection.
        if possible_vandalism:
            token_density = compute_avg_word_freq(text_curr)
            if token_density > TOKEN_DENSITY_LIMIT:
                return matched_words_prev, possible_vandalism
            else:
                possible_vandalism = False

        # Edit consists of adding new content, not changing/removing content
        if not text_prev:
            for sentence_curr, word in curr_slots:
                word_curr = Word()
                word_curr.value = word
                word_curr.token_id = self.token_id
                word_curr.origin_rev_id = self.revision_curr.id
                word_curr.last_rev_id = self.revision_curr.id
                sentence_curr.words.append(word_curr)
                self.token_id += 1
                self.revision_curr.original_adds += 1
                self.tokens.append(word_curr)
            return matched_words_prev, possible_vandalism

        full_texts = []

        def get_full_texts():
            if not full_texts:
                full_texts.append((
                    [word.value for word in iter_rev_tokens(self.revision_prev)],
                    split_into_tokens(self.text_curr),
                ))
            return full_texts[0]

        prev_for_curr, deleted_prev_indices = _match_word_sequences(
            text_prev,
            text_curr,
            get_full_texts=get_full_texts,
            prev_words=unmatched_words_prev,
        )
        for curr_index, prev_index in enumerate(prev_for_curr):
            sentence_curr, word = curr_slots[curr_index]
            if prev_index is None:
                word_curr = Word()
                word_curr.value = word
                word_curr.token_id = self.token_id
                word_curr.origin_rev_id = self.revision_curr.id
                word_curr.last_rev_id = self.revision_curr.id
                sentence_curr.words.append(word_curr)
                self.token_id += 1
                self.revision_curr.original_adds += 1
                self.tokens.append(word_curr)
            else:
                word_prev = unmatched_words_prev[prev_index]
                word_prev.matched = True
                sentence_curr.words.append(word_prev)
                matched_words_prev.append(word_prev)

        for prev_index in deleted_prev_indices:
            word_prev = unmatched_words_prev[prev_index]
            word_prev.matched = True
            word_prev.outbound.append(self.revision_curr.id)
            matched_words_prev.append(word_prev)

        return matched_words_prev, possible_vandalism

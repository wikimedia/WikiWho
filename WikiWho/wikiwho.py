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
    compute_avg_word_freq


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

# Caps nearest-neighbor recovery inside one SequenceMatcher replace/equal  opcode region. Higher values preserve more matches in broad edits but can  reintroduce expensive local scans.
WORD_MATCH_MAX_LOCAL_PAIRS = 10000

# Minimum positional drift allowed when deciding whether to reuse a previous Word object.
WORD_MATCH_MAX_DRIFT_MIN = 50

# Ratio-based positional drift allowed, computed against the larger unmatched side. Higher drift preserves longer moves; lower drift bounds cost and cross-section matches but loses lineage for content moved farther away.
WORD_MATCH_MAX_DRIFT_RATIO = 0.10


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


def _match_word_sequences(text_prev, text_curr):
    prev_for_curr = [None] * len(text_curr)

    prefix_len = _common_prefix_len(text_prev, text_curr)
    suffix_len = _common_suffix_len(text_prev, text_curr, prefix_len)
    for index in range(prefix_len):
        prev_for_curr[index] = index
    for index in range(suffix_len):
        prev_index = len(text_prev) - suffix_len + index
        curr_index = len(text_curr) - suffix_len + index
        prev_for_curr[curr_index] = prev_index

    prev_mid_start = prefix_len
    prev_mid_end = len(text_prev) - suffix_len
    curr_mid_start = prefix_len
    curr_mid_end = len(text_curr) - suffix_len
    prev_mid = text_prev[prev_mid_start:prev_mid_end]
    curr_mid = text_curr[curr_mid_start:curr_mid_end]

    if prev_mid and curr_mid:
        max_drift = _word_match_drift_limit(len(prev_mid), len(curr_mid))
        if _word_match_pair_estimate(prev_mid, curr_mid) <= WORD_MATCH_MAX_SEQUENCE_PAIRS:
            matcher = SequenceMatcher(None, prev_mid, curr_mid, autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal' and abs((prev_mid_start + i1) - (curr_mid_start + j1)) <= max_drift:
                    for prev_index, curr_index in zip(range(i1, i2), range(j1, j2)):
                        prev_for_curr[curr_mid_start + curr_index] = prev_mid_start + prev_index
                elif tag in ('replace', 'equal') and (i2 - i1) * (j2 - j1) <= WORD_MATCH_MAX_LOCAL_PAIRS:
                    local_matches = _nearest_word_matches(prev_mid[i1:i2],
                                                          curr_mid[j1:j2],
                                                          prev_mid_start + i1,
                                                          curr_mid_start + j1,
                                                          max_drift)
                    for curr_index, prev_index in local_matches.items():
                        prev_for_curr[curr_mid_start + j1 + curr_index] = prev_mid_start + i1 + prev_index
        else:
            local_matches = _nearest_word_matches(prev_mid, curr_mid,
                                                  prev_mid_start, curr_mid_start,
                                                  max_drift)
            for curr_index, prev_index in local_matches.items():
                prev_for_curr[curr_mid_start + curr_index] = prev_mid_start + prev_index

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

        prev_for_curr, deleted_prev_indices = _match_word_sequences(text_prev, text_curr)
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

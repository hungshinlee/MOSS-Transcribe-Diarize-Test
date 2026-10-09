"""Shared helpers for computing MER against MOSS-Transcribe-Diarize output,
used both by the training-time compute_metrics (finetune_with_eval.py) and
the post-hoc checkpoint/test-set evaluators (eval_hotword_testset.py,
select_checkpoint.py).
"""

import os
import re
import sys

from opencc import OpenCC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hotword_benchmark"))
from evaluate import (  # noqa: F401,E402  (re-exported)
    compute_mer,
    normalize_cjk_english_spacing,
    normalize_english_digit_words,
    normalize_numbers,
    normalize_spaced_letters,
    tokenize_mixed,
)

_TAG_RE = re.compile(r"\[\d+\.\d+\]|\[S\d+\]")
_SEGMENT_RE = re.compile(r"\[(\d+\.\d+)\]\[S(\d+)\](.*?)\[(\d+\.\d+)\]", re.DOTALL)
# Strip all punctuation except "/" -- training data uses "/" (e.g. "O2 3L/min")
# and we want to see whether the model actually learns to reproduce that
# format, so it must survive normalization while every other punctuation
# mark (which carries no signal for MER purposes) is dropped. \w matches CJK
# characters too, so this also drops CJK punctuation (，。！？「」etc.).
_PUNCT_RE = re.compile(r"[^\w\s/]", re.UNICODE)
_S2T = OpenCC("s2t")


def strip_tags(text: str) -> str:
    """Remove [start.xx]/[S01]-style timestamp+speaker tags so free text is
    comparable to untagged reference transcripts."""
    return _TAG_RE.sub("", text).strip()


def parse_tagged_segments(raw_text: str) -> list[dict]:
    """Parse raw (untagged-not-yet) '[start.xx][Sxx]text[end.xx]...' model
    output into a structured list of {"start","end","speaker","text"} dicts,
    for timestamp/speaker-accuracy scoring against ground-truth segments
    (AliMeeting/AISHELL-4 eval). Deliberately regex-based, not a strict state
    machine -- malformed/truncated trailing segments (e.g. generation cut off
    by max_new_tokens mid-segment) are simply not matched rather than raising,
    since a partial last segment carries no reliable end timestamp anyway."""
    return [
        {
            "start": float(m.group(1)),
            "speaker": f"S{m.group(2)}",
            "text": m.group(3),
            "end": float(m.group(4)),
        }
        for m in _SEGMENT_RE.finditer(raw_text)
    ]


def align_tokens(ref_tokens: list[str], hyp_tokens: list[str]) -> list[tuple]:
    """Standard Levenshtein alignment (unit cost per op), returning the full
    edit path as (op, ref_tok_or_None, hyp_tok_or_None) tuples in order,
    op in {"match","sub","del","ins"}. evaluate.py's own levenshtein_distance
    only returns the distance, not the alignment path -- this is needed
    separately so each edit operation can be attributed to a language bucket
    (see compute_mer_by_language)."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref_tokens[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp_tokens[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _classify_token(tok: str) -> str:
    if tok.isdigit():
        return "digit"
    if tok.isascii() and tok.isalpha():
        return "en"
    return "zh"


def compute_mer_by_language(ref_text: str, hyp_text: str, normalize: bool = True) -> dict[str, tuple[int, int]]:
    """Per-language MER: {"zh": (errors, ref_len), "en": (...), "digit": (...)}.

    A single Levenshtein alignment is run over the whole mixed token
    sequence, then each edit op is attributed to a language bucket by the
    REFERENCE token's language for match/sub/del, and by the HYPOTHESIS
    token's language for insertions (no ref token exists to classify by).
    This deliberately does NOT filter ref/hyp into independent per-language
    subsequences and diff each separately -- that would silently drop
    cross-language substitutions (e.g. ref="你" hyp="you": the "你"→missing
    look like a deletion in the zh-only subsequence, and "you"→missing looks
    like an insertion in the en-only subsequence with ref_len=0, so it gets
    thrown away as a (0,0) result instead of counted as one substitution
    error attributed to zh).

    Callers should accumulate the returned (errors, ref_len) tuples across
    an entire eval set and divide once at the end (same pattern as
    compute_mer's existing callers), so a single example with zero ref
    tokens in some language bucket doesn't lose insertion errors in that
    bucket -- the corpus-level denominator is what matters.
    """
    if normalize:
        ref_text = normalize_numbers(normalize_english_digit_words(normalize_spaced_letters(ref_text)))
        hyp_text = normalize_numbers(normalize_english_digit_words(normalize_spaced_letters(hyp_text)))
    ref_tokens = tokenize_mixed(ref_text)
    hyp_tokens = tokenize_mixed(hyp_text)
    ops = align_tokens(ref_tokens, hyp_tokens)

    errors = {"zh": 0, "en": 0, "digit": 0}
    ref_len = {"zh": 0, "en": 0, "digit": 0}
    for op, r, h in ops:
        if r is not None:
            ref_len[_classify_token(r)] += 1
        if op == "match":
            continue
        elif op in ("sub", "del"):
            errors[_classify_token(r)] += 1
        elif op == "ins":
            errors[_classify_token(h)] += 1
    return {lang: (errors[lang], ref_len[lang]) for lang in ("zh", "en", "digit")}


def normalize_for_eval(text: str, is_asr_output: bool = False) -> str:
    """Normalize ref/hyp text before MER comparison: strip ASR timestamp/
    speaker tags (hyp only), convert Simplified->Traditional (the finetune
    prompt is Simplified Chinese, so the base model can drift into Simplified
    output even though our reference transcripts are Traditional -- without
    this, that script difference gets counted as a content error), strip all
    punctuation except "/", then collapse CJK/English-digit boundary spaces
    (ASR tokenization artifact, not a content difference)."""
    if is_asr_output:
        text = strip_tags(text)
    text = _S2T.convert(text)
    text = _PUNCT_RE.sub("", text)
    text = normalize_cjk_english_spacing(text)
    return text.strip()

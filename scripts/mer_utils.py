"""Shared helpers for computing MER against MOSS-Transcribe-Diarize output,
used both by the training-time compute_metrics (finetune_with_eval.py) and
the post-hoc checkpoint/test-set evaluators (eval_hotword_testset.py,
select_checkpoint.py).
"""

import os
import re
import sys

import warnings

import cn2an
from opencc import OpenCC

# cn2an 遇到判斷不出來的中文數字片段（例如語音辨識輸出本身就不成規則的
# 「一二五十九」）會發 UserWarning 但照樣把該片段原樣保留繼續跑，不影響
# 正確性，只是很吵，這裡直接關掉。
warnings.filterwarnings("ignore", category=UserWarning, module="cn2an")

# ---------------------------------------------------------------------------
# 文字前處理 / 斷詞 / 編輯距離 / MER
# ---------------------------------------------------------------------------
CJK_CHAR_RE = re.compile(r"[一-鿿]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+")
DIGIT_RUN_RE = re.compile(r"[0-9]+")

_SPACED_LETTERS_RE = re.compile(r"(?<![A-Za-z])[A-Za-z](?:\s+[A-Za-z])+(?![A-Za-z])")
_CN_DECIMAL_POINT_RE = re.compile(r"(?<=\d)點(?=\d)")
_CN_NEGATIVE_RE = re.compile(r"負(?=\d)")
_CJK_ADJACENT_SPACE_RE = re.compile(
    r"(?<=[一-鿿])\s+(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])\s+(?=[一-鿿])"
)

_ENGLISH_DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_ENGLISH_DIGIT_WORD_RE = re.compile(
    r"\b(" + "|".join(_ENGLISH_DIGIT_WORDS) + r")\b", re.IGNORECASE
)


def preprocess_text(text: str) -> str:
    """去除標點/空白等非文字字元、轉小寫。熱詞的子字串比對用得到這個。"""
    if not text:
        return ""
    return re.sub(r"[^\w一-鿿]", "", text).lower()


def normalize_spaced_letters(text: str) -> str:
    """把用空白隔開的連續單字母（例如 "D C"、"B I D"）合併成一個詞
    （"DC"、"BID"）。這種空格通常是 ASR 把拼讀的縮寫拆開講，不是內容錯誤——
    只合併「連續兩個以上」的單字母，落單的字母（例如「A床」的 A）前後不是
    字母就不會比對到，維持原樣。"""
    return _SPACED_LETTERS_RE.sub(lambda m: re.sub(r"\s+", "", m.group()), text)


def normalize_english_digit_words(text: str) -> str:
    """把英文唸出來的個位數字（one/two/.../nine/zero）轉成阿拉伯數字，例如
    "SPO two" -> "SPO 2"，對應 SPO2 這種縮寫+數字唸法。只處理 0~9 單位數，
    不處理十位/複合數字。"""
    return _ENGLISH_DIGIT_WORD_RE.sub(lambda m: _ENGLISH_DIGIT_WORDS[m.group().lower()], text)


def normalize_numbers(text: str) -> str:
    """把文字裡的中文數字（含口語略寫，例如「五百八」= 580）轉成阿拉伯數字，
    非數字內容不動。cn2an 的 transform() 本身不處理小數點/負號，額外用兩條
    只在「緊鄰數字」時才生效的規則補上「七點零」->"7.0"、「負二」->"-2"，
    避免誤傷「地點」「負責」這類正常詞（前後不是數字就不會被比對到）。
    轉換失敗（cn2an 對這段文字判斷不出來）時，原樣保留整段文字。"""
    try:
        converted = cn2an.transform(text, "cn2an")
    except Exception:
        return text
    converted = _CN_DECIMAL_POINT_RE.sub(".", converted)
    converted = _CN_NEGATIVE_RE.sub("-", converted)
    return converted


def normalize_cjk_english_spacing(text: str) -> str:
    """去掉中文字跟英文字母/數字交界處的空白（例如「X 光」-> 「X光」）。這種
    空格通常是 ASR 在中英文交界處插入的分詞產物，不是內容差異。只處理「中文
    字」緊鄰「英文字母/數字」的位置，純英文片語內部本來就有意義的空格（例如
    「normal saline」）不會被合併，因為前後都不是中文字，比對不到。"""
    return _CJK_ADJACENT_SPACE_RE.sub("", text)


def tokenize_mixed(text: str) -> list[str]:
    """把中英數混合文字切成 token：中文字＝1 token，連續英文字母＝1 token，
    連續數字＝1 token，標點/空白略過不產生 token。MER 在這個 token 序列上算編輯
    距離，讓「AMMONIA 整段唸錯」只算 1 個錯誤，不會因為英文單字比字元長就被
    放大成好幾個錯誤，也從根本上排除標點符號對指標的影響。"""
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if CJK_CHAR_RE.match(ch):
            tokens.append(ch.lower())
            i += 1
            continue
        word_match = ENGLISH_WORD_RE.match(text, i)
        if word_match:
            tokens.append(word_match.group().lower())
            i = word_match.end()
            continue
        digit_match = DIGIT_RUN_RE.match(text, i)
        if digit_match:
            tokens.append(digit_match.group())
            i = digit_match.end()
            continue
        i += 1
    return tokens


def levenshtein_distance(s1, s2) -> int:
    """通用編輯距離，s1/s2 可以是字串或任意序列（例如 token list）。"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_mer(ref_text: str, hyp_text: str, normalize: bool = True) -> tuple[int, int]:
    """回傳 (編輯距離, 參考 token 數)。中文字/英文單字/數字各當一個 token 算，
    標點/空白忽略。參考 token 數為 0 時視為無法計算（不列入整體統計分母）。

    normalize=True（預設）時，比對前會先合併空格拼字母、把中英文數字都轉成
    阿拉伯數字（見 normalize_spaced_letters()/normalize_english_digit_words()/
    normalize_numbers()），避免純格式差異（"D C" vs "DC"、"五百八" vs "580"、
    "SPO two" vs "SPO2"）被當成內容錯誤。設成 False 可以看正規化前的原始
    比對結果。"""
    if normalize:
        ref_text = normalize_numbers(normalize_english_digit_words(normalize_spaced_letters(ref_text)))
        hyp_text = normalize_numbers(normalize_english_digit_words(normalize_spaced_letters(hyp_text)))
    ref_tokens = tokenize_mixed(ref_text)
    hyp_tokens = tokenize_mixed(hyp_text)
    if not ref_tokens:
        return 0, 0
    return levenshtein_distance(ref_tokens, hyp_tokens), len(ref_tokens)


# ---------------------------------------------------------------------------
# 熱詞命中判定輔助
# ---------------------------------------------------------------------------
def count_keyword_occurrences(keyword: str, text: str, sibling_supersets: list[str] | None = None) -> int:
    """計算 keyword 在 text 裡出現幾次（忽略大小寫、忽略中英文交界空白、不重疊比對）。"""
    masked_text = text
    for superset in sibling_supersets or []:
        masked_text = re.sub(
            re.escape(superset), lambda m: " " * len(m.group()), masked_text, flags=re.IGNORECASE
        )
    normalized_keyword = normalize_cjk_english_spacing(keyword)
    normalized_text = normalize_cjk_english_spacing(masked_text)
    return len(re.findall(re.escape(normalized_keyword), normalized_text, re.IGNORECASE))


def keyword_present(keyword: str, text: str, sibling_supersets: list[str] | None = None) -> bool:
    """檢查 keyword 是否以子字串形式出現在 text 裡至少一次（忽略大小寫）。"""
    return count_keyword_occurrences(keyword, text, sibling_supersets) > 0


def build_sibling_superset_map(hotwords_map: dict) -> dict[str, list[str]]:
    """對整個 benchmark 的熱詞詞彙表（不只單一音檔），找出每個熱詞的「較長親戚」。"""
    all_keywords = sorted({kw for kws in hotwords_map.values() for kw in kws})
    lowered = {kw: kw.lower() for kw in all_keywords}
    supersets: dict[str, list[str]] = {kw: [] for kw in all_keywords}
    for a in all_keywords:
        for b in all_keywords:
            if a == b or lowered[a] == lowered[b]:
                continue
            if lowered[a] in lowered[b] and len(lowered[a]) < len(lowered[b]):
                supersets[a].append(b)
    return supersets


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
    op in {"match","sub","del","ins"}. levenshtein_distance() only returns
    the distance, not the alignment path -- this is needed separately so
    each edit operation can be attributed to a language bucket
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

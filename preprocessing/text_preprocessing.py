"""
Sentiment Analysis – Advanced Text Preprocessing Pipeline v2.0

Steps (order matters):
  1.  None / type guard
  2.  HTML tag removal (BeautifulSoup)
  3.  URL normalisation  → __URL__
  4.  Unicode / encoding normalisation
  5.  Contraction expansion
  6.  Lowercase
  7.  Emoji / emoticon normalisation
  8.  Negation scope marking  (not_good → not_good)
  9.  Repeated character normalisation  (looooove → love)
  10. Special-character removal (keep apostrophes)
  11. Tokenisation
  12. Selective stop-word removal  (negation words PRESERVED)
  13. Lemmatisation
  14. Empty-text detection / fallback

IMPORTANT:
  - Negation words are NEVER removed.
  - Sentiment-bearing intensifiers (very, extremely, too …) are PRESERVED.
  - Contrastive conjunctions (but, although …) are PRESERVED.
  - Identical logic used at training time AND inference time.
"""

from __future__ import annotations

import re
import os
import unicodedata
import logging
from typing import Union

import nltk

# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------
_NLTK_DATA_DIR = os.environ.get(
    "NLTK_DATA", os.path.join(os.path.expanduser("~"), "nltk_data")
)
if not os.access(os.path.expanduser("~"), os.W_OK):
    _NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(_NLTK_DATA_DIR, exist_ok=True)
if _NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DATA_DIR)

_NLTK_RESOURCES = [
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
    ("corpora/wordnet",      "wordnet"),
    ("corpora/omw-1.4",      "omw-1.4"),
]


def _ensure_nltk() -> None:
    for path, pkg in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True, download_dir=_NLTK_DATA_DIR)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Could not download NLTK resource %s: %s", pkg, exc
                )


_ensure_nltk()

from nltk.corpus   import stopwords as _sw_corpus
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negation words – MUST be preserved
# ---------------------------------------------------------------------------
NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor", "nobody",
    "nothing", "nowhere", "none", "hardly", "barely",
    "scarcely", "doesn't", "doesnt", "isn't", "isnt",
    "wasn't", "wasnt", "aren't", "arent",
    "won't", "wont", "wouldn't", "wouldnt",
    "don't", "dont", "didn't", "didnt",
    "haven't", "havent", "hasn't", "hasnt",
    "hadn't", "hadnt", "shouldn't", "shouldnt",
    "couldn't", "couldnt", "mustn't", "mustnt",
    "cannot", "can't", "cant", "needn't", "neednt",
    "without", "lack", "lacking", "absent", "absence",
    "n't",
})

# Sentiment intensifiers – MUST be preserved
INTENSIFIERS: frozenset[str] = frozenset({
    "very", "extremely", "absolutely", "incredibly", "utterly",
    "totally", "completely", "highly", "really", "quite", "rather",
    "too", "so", "such", "deeply", "terribly", "awfully",
    "exceptionally", "remarkably", "tremendously", "super", "ultra",
    "badly", "strongly", "thoroughly", "genuinely", "truly",
})

# Contrastive conjunctions – preserve context
CONTRASTIVES: frozenset[str] = frozenset({
    "but", "however", "although", "though", "despite",
    "yet", "nevertheless", "nonetheless", "while", "whereas",
    "even", "still", "except", "unless",
})

# Words to always keep (negation + intensifiers + contrastives)
_ALWAYS_KEEP: frozenset[str] = NEGATION_WORDS | INTENSIFIERS | CONTRASTIVES

# ---------------------------------------------------------------------------
# Stop-word set  (NLTK English minus always-keep words)
# ---------------------------------------------------------------------------
_RAW_STOPWORDS: set[str] = set(_sw_corpus.words("english"))
STOP_WORDS: frozenset[str] = frozenset(_RAW_STOPWORDS - _ALWAYS_KEEP)

# ---------------------------------------------------------------------------
# Lemmatiser (singleton)
# ---------------------------------------------------------------------------
_LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Contraction map
# ---------------------------------------------------------------------------
_CONTRACTIONS: dict[str, str] = {
    "i'm": "i am",       "i've": "i have",     "i'll": "i will",
    "i'd": "i would",    "you're": "you are",   "you've": "you have",
    "you'll": "you will","you'd": "you would",   "he's": "he is",
    "she's": "she is",   "it's": "it is",        "we're": "we are",
    "we've": "we have",  "we'll": "we will",     "we'd": "we would",
    "they're": "they are","they've": "they have","they'll": "they will",
    "they'd": "they would","that's": "that is",  "that'll": "that will",
    "who's": "who is",   "what's": "what is",    "here's": "here is",
    "there's": "there is","where's": "where is", "how's": "how is",
    # Negations – expand carefully to keep "not"
    "isn't": "is not",   "aren't": "are not",   "wasn't": "was not",
    "weren't": "were not","won't": "will not",   "wouldn't": "would not",
    "don't": "do not",   "doesn't": "does not", "didn't": "did not",
    "haven't": "have not","hasn't": "has not",  "hadn't": "had not",
    "shouldn't": "should not","couldn't": "could not","mustn't": "must not",
    "can't": "cannot",   "needn't": "need not", "mightn't": "might not",
    "shan't": "shall not","let's": "let us",
}

# ---------------------------------------------------------------------------
# Emoji / emoticon map
# ---------------------------------------------------------------------------
_EMOTICONS: dict[str, str] = {
    r":-?\)": " positiveemoji ",  r":-?\(": " negativeemoji ",
    r":-?D": " positiveemoji ",   r":-?P": " positiveemoji ",
    r":-?/": " neutralemoji ",    r":-?\|": " neutralemoji ",
    r";-?\)": " positiveemoji ",  r":\)": " positiveemoji ",
    r":\(": " negativeemoji ",    r":D": " positiveemoji ",
    r"</3": " negativeemoji ",    r"<3": " positiveemoji ",
    r"xo+": " positiveemoji ",    r":\'+\(": " negativeemoji ",
}

# Pre-compile emoticon patterns
_EMOTICON_RE = [(re.compile(pat, re.IGNORECASE), rep)
                for pat, rep in _EMOTICONS.items()]

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------
_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_URL_RE         = re.compile(r"https?://\S+|www\.\S+|ftp://\S+")
_EMAIL_RE       = re.compile(r"\S+@\S+\.\S+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")          # 3+ repeated chars → 2
_MULTI_SPACE_RE = re.compile(r"\s+")
_SPECIAL_RE     = re.compile(r"[^a-z\s']")          # keep letters, spaces, apostrophes
_NEGATION_RE    = re.compile(
    r"\b(not|never|no|n't|neither|nor|hardly|barely|scarcely|cannot|without)\b"
    r"\s+(\w+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_contractions(text: str) -> str:
    """Expand English contractions using a fixed dictionary."""
    for contraction, expanded in _CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b",
                      expanded, text, flags=re.IGNORECASE)
    return text


def normalise_emoticons(text: str) -> str:
    """Replace common emoticons with sentiment tokens."""
    for pattern, replacement in _EMOTICON_RE:
        text = pattern.sub(replacement, text)
    return text


def normalise_unicode(text: str) -> str:
    """NFKD normalise and drop combining characters (accents etc.)."""
    normalised = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalised if not unicodedata.combining(ch))


def mark_negation_scope(text: str) -> str:
    """
    Attach 'not_' prefix to the word immediately following a negation word.
    Example: 'not good' → 'not not_good'
    This helps TF-IDF capture negation bigrams as features.
    """
    return _NEGATION_RE.sub(
        lambda m: m.group(1) + " not_" + m.group(2), text
    )


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline.  Returns a cleaned, lemmatised string
    ready for TF-IDF vectorisation.

    Safe to call with None or empty strings – returns "".
    """
    # --- guard ---
    if not text or not isinstance(text, str):
        return ""

    # 1. HTML removal
    if _BS4_AVAILABLE:
        try:
            text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        except Exception:
            text = _HTML_TAG_RE.sub(" ", text)
    else:
        text = _HTML_TAG_RE.sub(" ", text)

    # 2. URL and email replacement
    text = _URL_RE.sub(" url ", text)
    text = _EMAIL_RE.sub(" email ", text)

    # 3. Unicode normalisation
    text = normalise_unicode(text)

    # 4. Emoticon normalisation (before lowercasing to catch :D etc.)
    text = normalise_emoticons(text)

    # 5. Lowercase
    text = text.lower()

    # 6. Contraction expansion
    text = expand_contractions(text)

    # 7. Negation scope marking
    text = mark_negation_scope(text)

    # 8. Repeated character normalisation  (looove → loove)
    text = _REPEAT_CHAR_RE.sub(r"\1\1", text)

    # 9. Special-character removal (keep apostrophes for n't, 's etc.)
    text = _SPECIAL_RE.sub(" ", text)

    # 10. Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    # 11. Tokenise
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 12. Stop-word removal (preserve _ALWAYS_KEEP words)
    # 13. Lemmatise
    cleaned_tokens: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        base = tok.lower()
        if base in _ALWAYS_KEEP:
            cleaned_tokens.append(base)
        elif base not in STOP_WORDS and len(base) > 1:
            cleaned_tokens.append(_LEMMATIZER.lemmatize(base))

    result = " ".join(cleaned_tokens)
    return result if result else ""


def preprocess_series(series) -> "pd.Series":
    """
    Vectorised preprocessing of a pandas Series.
    Returns a new Series of cleaned strings.
    """
    import pandas as pd  # local import to avoid hard dependency at module level
    return series.fillna("").astype(str).map(clean_text)

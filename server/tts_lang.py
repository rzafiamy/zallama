"""Pick a Kokoro voice from the language of the text to synthesize.

kokoro-server has no language argument of its own: it derives the phonemizer
language from the *voice* prefix (`ff_siwis` → `fr-FR`, `af_heart` → `en-US`,
...). So a French sentence sent with the default English voice is phonemized
with English sounds — "Ensuite" comes out as `ɛnsuːtiː` instead of `ɑ̃syit`.
When the caller does not say which voice it wants, guessing the language from
the text is the only way to avoid that.

This is a deliberately small heuristic, not a language identifier: it only
separates the languages Kokoro actually ships voices for, and it answers `None`
whenever the evidence is thin (short inputs especially) so the caller can fall
back to its configured default rather than act on a coin flip.
"""
from __future__ import annotations

import re

# One representative voice per language Kokoro supports. These are the defaults
# used when the language was detected but no voice was requested; every other
# voice stays reachable by asking for it explicitly.
VOICE_BY_LANGUAGE: dict[str, str] = {
    "en": "af_heart",
    "fr": "ff_siwis",
    "es": "ef_dora",
    "it": "if_sara",
    "pt": "pf_dora",
    "hi": "hf_alpha",
    "ja": "jf_alpha",
    "zh": "zf_xiaoxiao",
}

# Common function words, chosen to be frequent enough to show up in a sentence
# or two. Single-letter words ("a", "o", "e", "y", "i") are left out on purpose:
# they collide across these languages and with English articles, so they cost
# more accuracy than they buy. Overlap between the Romance sets is expected —
# the decision is made on the total, not on any one word.
_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "the", "and", "is", "are", "was", "were", "of", "to", "in", "that",
        "it", "for", "with", "you", "this", "on", "be", "have", "has", "not",
        "but", "from", "they", "we", "what", "which", "will", "would", "can",
        "there", "their", "about", "when", "your", "all",
    }),
    "fr": frozenset({
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "est",
        "sont", "dans", "que", "qui", "pour", "pas", "ce", "cette", "sur",
        "avec", "vous", "nous", "plus", "mais", "tout", "comme", "être",
        "aux", "son", "ses", "leur", "très", "alors", "donc", "aussi",
        "sans", "elle", "ils", "cela", "peut", "faire",
    }),
    "es": frozenset({
        "el", "los", "las", "una", "unos", "del", "es", "son", "que", "por",
        "para", "con", "más", "pero", "como", "esta", "este", "hay", "muy",
        "cuando", "porque", "todo", "sus", "sin", "también", "ser", "hacer",
    }),
    "it": frozenset({
        "il", "lo", "gli", "una", "del", "della", "di", "è", "sono", "che",
        "per", "con", "non", "più", "questo", "questa", "anche", "molto",
        "quando", "perché", "come", "sua", "senza", "essere", "fare",
    }),
    "pt": frozenset({
        "os", "as", "uma", "do", "da", "dos", "das", "é", "são", "que", "em",
        "por", "para", "com", "não", "mais", "mas", "como", "este", "esta",
        "tudo", "muito", "quando", "porque", "você", "sem", "ser", "fazer",
    }),
}

# Characters that only one of these languages really uses. Worth more than a
# single word hit, but counted once however often they appear so a text full of
# "ç" cannot drown out everything else.
_DIACRITIC_HINTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("es", frozenset("ñ¿¡")),
    ("pt", frozenset("ãõ")),
    ("fr", frozenset("œêîûëï")),
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Scripts that identify a language outright, no word list needed.
_KANA = (range(0x3040, 0x30A0), range(0x30A0, 0x3100))
_HAN = range(0x4E00, 0xA000)
_DEVANAGARI = range(0x0900, 0x0980)


def _script_language(text: str) -> str | None:
    """Return a language fixed by the writing system, or None for Latin text."""
    kana = han = deva = 0
    for ch in text:
        cp = ord(ch)
        if any(cp in r for r in _KANA):
            kana += 1
        elif cp in _HAN:
            han += 1
        elif cp in _DEVANAGARI:
            deva += 1
    if deva >= 2:
        return "hi"
    # Japanese mixes kana with Han; any meaningful kana presence settles it.
    if kana >= 2:
        return "ja"
    if han >= 2:
        return "zh"
    return None


def detect_language(text: str) -> str | None:
    """Best-effort language of `text`, or None when the evidence is too thin.

    Returning None is a normal outcome, not a failure: it means the caller
    should keep whatever default it already had.
    """
    if not text:
        return None
    if script := _script_language(text):
        return script

    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < 3:
        # A handful of words is not enough to tell "Merci" from "Mercy".
        return None

    scores = {lang: sum(w in markers for w in words) for lang, markers in _MARKERS.items()}
    lowered = text.lower()
    for lang, chars in _DIACRITIC_HINTS:
        if any(c in lowered for c in chars):
            scores[lang] += 2

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best_lang, best), (_, runner_up) = ranked[0], ranked[1]
    # Require both an absolute floor and a clear margin: Romance languages share
    # too many function words for a one-hit lead to mean anything.
    if best < 2 or best == runner_up:
        return None
    return best_lang


def voice_for_text(text: str, fallback: str | None = None) -> str | None:
    """Voice to synthesize `text` with, or `fallback` when unsure."""
    lang = detect_language(text)
    if lang is None:
        return fallback
    return VOICE_BY_LANGUAGE.get(lang, fallback)

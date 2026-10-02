"""Text normalization in front of the TTS engines.

TTS models read digits, symbols and Markdown poorly (Pocket TTS turns "9h30"
into noise; Kokoro reads simple numbers but not "1 250 000 €"; chat answers
are full of lists and bold). A TTS entry opts in with:

    params:
      normalizer: tn                 # a `normalization` model (tn-server)
      normalizer_llm: gemma-e2b-tn   # optional `text` model for leftovers
      language: fr                   # when the route can't tell

Chain, per request:
1. tn-server: lexicon, then rules. With an LLM configured, in *safe* mode:
   numbers the rules can't read for sure (codes, phone chains, "(555)")
   stay as digits, everything else is spelled out exactly.
2. The LLM, only on sentences that still contain digits, Roman numerals or
   abbreviations, all at once (llama-server slots run them in parallel),
   few-shot, temperature 0. The LLM never sees the amounts the rules
   already read, so it cannot change them.
3. Guard: an LLM answer that lost plain words of its sentence (or is empty
   or runaway) is replaced by tn's strict reading of that sentence.

Benchmark behind these choices (16 hand-referenced FR/EN sentences):
tn strict 6.0 % WER; tn safe + Gemma-4-E2B QAT 1.0 %, 0 dropped words,
~190 ms per sentence needing the LLM (tn-rs docs/llm-pass.md).
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable

import httpx

logger = logging.getLogger("zallama.tts_normalize")

# What the rules leave to the LLM: digits, Roman numerals (2+ capitals,
# alone or with an ordinal suffix: "XIV", "XVIIe", "VIIIth"; "Ier"), and
# short capitalized abbreviations inside a sentence ("St. James", "Oct. 21").
# Single capitals are not Roman numerals here: "Le", "Ce", "I".
NEEDS_LLM = re.compile(
    r"[0-9]"
    r"|\b[IVXLCDM]{2,}(?:e|er|re|ème|th|st|nd|rd)?\b"
    r"|\bIe?re?\b(?=\s)"
    r"|\b[A-Z][a-z]{0,2}\.(?=\s+\S)"
)
# Sentence boundary after tn's output (one line = one sentence, joined by spaces).
_SENTENCE = re.compile(r"(?<=[.!?…:;])\s+")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*\.?")

SYSTEM = (
    "You are a text normalizer for a text-to-speech engine. Rewrite the user's text exactly as it "
    "should be spoken aloud, in the same language: spell out numbers, dates, times, amounts with "
    "currency, percentages, temperatures, units, ordinals, Roman numerals, phone numbers and "
    "abbreviations; remove Markdown symbols. Keep every other word, in the same order, with the same "
    "punctuation. Output only the rewritten text."
)
SHOTS = [
    ("Le train part à 7h45 du quai 3, billet à 23,50 €.",
     "Le train part à sept heures quarante-cinq du quai trois, billet à vingt-trois euros cinquante."),
    ("The 2nd edition sold 15,000 copies in 2019 at $9.99 each.",
     "The second edition sold fifteen thousand copies in twenty nineteen at nine dollars and ninety-nine cents each."),
    ("**Rappel** : M. Leroy arrive le 12/03 à 16h.",
     "Rappel : Monsieur Leroy arrive le douze mars à seize heures."),
]


def dropped_words(source: str, output: str) -> list[str]:
    """Plain words of `source` missing from `output`.

    Abbreviations ("Mme", "St."), all-caps tokens (Roman numerals, acronyms)
    and one-letter words are expected to change and are not counted.
    """
    out_words = {w.lower().rstrip(".") for w in _WORD.findall(output.replace("-", " "))}
    missing = []
    for tok in _WORD.findall(source.replace("-", " ")):
        word = tok.rstrip(".")
        abbreviation = (tok.endswith(".") and len(word) <= 4) or (
            word[0].isupper() and len(word) <= 4 and not word.isupper() and word[1:].islower()
            and word.lower() in _ABBREVIATIONS
        )
        roman = re.fullmatch(r"[IVXLCDM]+(?:e|er|re|ème|th|st|nd|rd)?", word) is not None
        if len(word) <= 1 or word.isupper() or abbreviation or roman:
            continue
        if word.lower() not in out_words:
            missing.append(word)
    return missing


# Abbreviations written without a period that the LLM rightly expands.
_ABBREVIATIONS = {"mme", "mmes", "mlle", "mlles", "dr", "pr", "st", "ste", "me", "mgr"}


Resolve = Callable[..., Awaitable]


async def normalize_for_tts(
    text: str,
    language: str | None,
    normalizer: str,
    llm: str | None,
    *,
    resolve: Resolve,
    pm,
    registry,
    timeout: float = 60.0,
) -> tuple[str, str]:
    """Normalized `text` and a short description of what ran
    ("rules", "rules+llm 3/3", ...)."""
    lang = language or "other"
    tn = await resolve(normalizer, pm, registry, endpoint="normalize")
    tn.touch()

    async def tn_call(payload: dict) -> dict:
        async with pm.serving(tn, "normalize"):
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{tn.base_url}/v1/normalize", json=payload)
        resp.raise_for_status()
        return resp.json()

    mode = "safe" if llm else "strict"
    out = (await tn_call({"text": text, "language": lang, "mode": mode}))["text"]
    if not llm:
        return out, "rules"

    sentences = _SENTENCE.split(out)
    todo = [i for i, s in enumerate(sentences) if NEEDS_LLM.search(s)]
    if not todo:
        return out, "rules"

    model = await resolve(llm, pm, registry, endpoint="chat/completions")
    model.touch()
    messages = [{"role": "system", "content": SYSTEM}]
    for src, dst in SHOTS:
        messages += [{"role": "user", "content": src}, {"role": "assistant", "content": dst}]

    async def ask(client: httpx.AsyncClient, sentence: str) -> str:
        body = {
            "model": llm,
            "temperature": 0,
            "max_tokens": 64 + 2 * len(sentence),
            "messages": messages + [{"role": "user", "content": sentence}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await client.post(f"{model.base_url}/v1/chat/completions", json=body)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
        return content.strip().strip('"«»“”').strip()

    async with pm.serving(model, "chat/completions"):
        async with httpx.AsyncClient(timeout=timeout) as client:
            answers = await asyncio.gather(
                *(ask(client, sentences[i]) for i in todo), return_exceptions=True
            )

    fallback = []
    for i, answer in zip(todo, answers):
        src = sentences[i]
        if isinstance(answer, Exception):
            logger.warning("normalizer LLM failed on %r: %s", src[:80], answer)
            fallback.append(i)
        elif not answer or len(answer) > 4 * len(src) + 80 or dropped_words(src, answer):
            logger.info("normalizer LLM answer rejected for %r: %r", src[:80], answer[:120])
            fallback.append(i)
        else:
            sentences[i] = answer
    if fallback:
        strict = await tn_call(
            {"texts": [sentences[i] for i in fallback], "language": lang, "mode": "strict"}
        )
        for i, s in zip(fallback, strict["texts"]):
            sentences[i] = s
    return " ".join(sentences), f"rules+llm {len(todo) - len(fallback)}/{len(todo)}"

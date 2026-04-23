from __future__ import annotations

import re


# Minimal starter list. Extend as needed for your demo/institution rules.
_TABOO_WORDS = {
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "dick",
    "pussy",
    "cunt",
    "sex",
    "porn",
    "nude",
    "naked",
    "dildo",
    "bastard",
    "whore",
    "slut",
    "gay",
    "lesbian",
    "transgender",
    "bisexual",
    "pansexual",
    "queer",
    "intersex",
    "asexual",
    "agender",
    "genderfluid",
}


def _compile_obfuscated_patterns(words: set[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for w in sorted(words, key=len, reverse=True):
        # Match common obfuscations like f*u*c*k or f.u.c.k (allow non-alnum between letters).
        body = r"[\W_]*".join(map(re.escape, w))
        patterns.append(re.compile(rf"\b{body}\b", flags=re.IGNORECASE))
    return patterns


_OBFUSCATED_PATTERNS = _compile_obfuscated_patterns(_TABOO_WORDS)
_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def contains_taboo(text: str) -> bool:
    if not text:
        return False

    lowered = text.casefold()
    tokens = set(_TOKEN_RE.findall(lowered))
    if tokens & _TABOO_WORDS:
        return True

    # Catch simple punctuation-separated obfuscations.
    return any(p.search(text) is not None for p in _OBFUSCATED_PATTERNS)


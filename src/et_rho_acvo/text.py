from __future__ import annotations

import re
from typing import Iterable


_WORD_RE = re.compile(r"(?u)\b\w+\b")


def tokenize_text(text: str, language: str = "en", min_token_length: int = 1) -> list[str]:
    value = "" if text is None else str(text)
    if language.lower().startswith("th"):
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError as exc:
            raise RuntimeError("Thai tokenization requires PyThaiNLP. Install the project dependencies.") from exc
        tokens = word_tokenize(value, engine="newmm", keep_whitespace=False)
    else:
        tokens = _WORD_RE.findall(value.lower())
    return [token for token in tokens if len(token.strip()) >= min_token_length and token.strip()]


def tokenize_documents(documents: Iterable[str], language: str, min_token_length: int = 1) -> list[list[str]]:
    return [tokenize_text(doc, language, min_token_length) for doc in documents]

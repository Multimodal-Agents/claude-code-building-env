"""
text_chunker.py — Sentence-level text splitting for TTS streaming.

Public API
----------
class TextChunker
    clean(text) -> str
    chunk(text) -> list[str]

class StreamingChunker
    feed(token) -> Iterator[str]
    flush()     -> str | None
"""

import re
from typing import Iterator


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Markdown noise to strip before TTS (preserves punctuation & alphanumeric)
STRIP_MD = re.compile(r"[*#`_~\[\]{}|\\<>@]")

# Sentence-boundary split rules (applied in order):
#   1. After sentence-ending punctuation followed by whitespace
#   2. Double newline (paragraph break)
#   3. Numbered list item start on its own line
SPLITS = re.compile(
    r"(?<=[.!?])\s+"          # after .!? + whitespace
    r"|(?=\n\n)"              # before paragraph break
    r"|(?<=\n)(?=\d+\.\s)"   # before numbered list line
)


# ---------------------------------------------------------------------------

class TextChunker:
    """
    Splits a complete text string into TTS-friendly chunks.

    Parameters
    ----------
    min_chars : Chunks shorter than this are merged with the next one.
    max_chars : Chunks longer than this are hard-split at the nearest space.
    """

    def __init__(self, min_chars: int = 15, max_chars: int = 250) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def clean(self, text: str) -> str:
        """Strip markdown special characters, collapse whitespace."""
        text = STRIP_MD.sub("", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def chunk(self, text: str) -> list[str]:
        """
        Split *text* into a list of clean TTS-suitable sentences/phrases.

        - Strips markdown noise.
        - Splits on sentence boundaries.
        - Merges short fragments with the next chunk.
        - Hard-splits chunks that exceed max_chars.
        """
        text = self.clean(text)
        if not text:
            return []

        raw_parts = [p.strip() for p in SPLITS.split(text) if p.strip()]
        merged: list[str] = []

        for part in raw_parts:
            if merged and len(merged[-1]) < self.min_chars:
                merged[-1] = (merged[-1] + " " + part).strip()
            else:
                merged.append(part)

        # Hard-split oversized chunks
        final: list[str] = []
        for chunk in merged:
            while len(chunk) > self.max_chars:
                split_at = chunk.rfind(" ", 0, self.max_chars)
                if split_at == -1:
                    split_at = self.max_chars
                final.append(chunk[:split_at].strip())
                chunk = chunk[split_at:].strip()
            if chunk:
                final.append(chunk)

        return [c for c in final if c]


# ---------------------------------------------------------------------------

class StreamingChunker:
    """
    Incrementally buffers a token stream and yields complete TTS chunks.

    Usage::

        sc = StreamingChunker()
        for token in llm_stream:
            for chunk in sc.feed(token):
                tts_queue.push(chunk)
        tail = sc.flush()
        if tail:
            tts_queue.push(tail)
    """

    # Sentence-end detection (used for boundary detection on the running buffer)
    _SENTENCE_END = re.compile(r"[.!?]\s")
    _PARAGRAPH    = re.compile(r"\n\n")

    def __init__(self, min_chars: int = 15, max_chars: int = 250) -> None:
        self._chunker = TextChunker(min_chars=min_chars, max_chars=max_chars)
        self._buf = ""

    def feed(self, token: str) -> Iterator[str]:
        """
        Consume *token* and yield any complete chunks that have formed.

        A chunk is emitted when the buffer contains a sentence boundary AND
        the text before that boundary is at least ``min_chars`` long.
        """
        self._buf += token

        while True:
            # Find the earliest sentence-end or paragraph boundary in buffer
            m_sent = self._SENTENCE_END.search(self._buf)
            m_para = self._PARAGRAPH.search(self._buf)

            candidates = [m for m in (m_sent, m_para) if m is not None]
            if not candidates:
                break

            m = min(candidates, key=lambda x: x.end())
            boundary = m.end()

            segment = self._buf[:boundary].strip()
            if len(segment) >= self._chunker.min_chars:
                self._buf = self._buf[boundary:]
                chunks = self._chunker.chunk(segment)
                yield from chunks
            else:
                # Not long enough yet — wait for more tokens
                break

    def flush(self) -> str | None:
        """Return (and clear) any remaining buffered text."""
        remainder = self._buf.strip()
        self._buf = ""
        if remainder:
            cleaned = self._chunker.clean(remainder)
            return cleaned if cleaned else None
        return None

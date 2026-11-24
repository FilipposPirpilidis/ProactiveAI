import asyncio
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "with", "you",
    "αλλα", "απο", "αυτο", "για", "δεν", "ειναι", "ενα", "εχει", "η", "θα",
    "και", "με", "μια", "να", "ο", "οι", "που", "πως", "σε", "στη", "στην",
    "στο", "στον", "τα", "την", "της", "τι", "το", "τον", "των",
}


@dataclass(frozen=True)
class KnowledgeSnippet:
    source: str
    heading: str
    content: str
    score: float


@dataclass(frozen=True)
class _IndexedChunk:
    source: str
    heading: str
    content: str
    terms: Counter[str]


class MarkdownKnowledgeBase:
    """Small, dependency-free retrieval index for trusted local Markdown files."""

    def __init__(
        self,
        root_path: str,
        *,
        enabled: bool = True,
        result_limit: int = 4,
        chunk_characters: int = 1_400,
        max_context_characters: int = 5_000,
        max_file_bytes: int = 1_000_000,
        refresh_seconds: float = 30.0,
    ) -> None:
        self.root_path = Path(root_path)
        self.enabled = enabled
        self.result_limit = result_limit
        self.chunk_characters = chunk_characters
        self.max_context_characters = max_context_characters
        self.max_file_bytes = max_file_bytes
        self.refresh_seconds = refresh_seconds
        self._chunks: tuple[_IndexedChunk, ...] = ()
        self._document_frequency: Counter[str] = Counter()
        self._file_count = 0
        self._last_refresh = 0.0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self.enabled:
            await self.refresh(force=True)

    async def refresh(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_refresh < self.refresh_seconds:
            return
        async with self._lock:
            now = time.monotonic()
            if not force and now - self._last_refresh < self.refresh_seconds:
                return
            chunks, file_count = await asyncio.to_thread(self._build_index)
            document_frequency: Counter[str] = Counter()
            for chunk in chunks:
                document_frequency.update(chunk.terms.keys())
            self._chunks = tuple(chunks)
            self._document_frequency = document_frequency
            self._file_count = file_count
            self._last_refresh = now

    async def search(self, query: str) -> tuple[KnowledgeSnippet, ...]:
        if not self.enabled:
            return ()
        await self.refresh()
        query_terms = set(_terms(query[-8_000:]))
        if not query_terms or not self._chunks:
            return ()

        chunk_count = len(self._chunks)
        ranked: list[KnowledgeSnippet] = []
        for chunk in self._chunks:
            shared = query_terms & chunk.terms.keys()
            if not shared:
                continue
            # One weak/common word is not enough to inject an unrelated file.
            distinctive = {
                term
                for term in shared
                if len(term) >= 5
                and (
                    chunk_count <= 3
                    or self._document_frequency[term] / chunk_count <= 0.35
                )
            }
            if len(shared) < 2 and not distinctive:
                continue
            score = sum(
                (1.0 + math.log1p(chunk.terms[term]))
                * math.log((chunk_count + 1) / (self._document_frequency[term] + 0.5))
                for term in shared
            )
            metadata_terms = set(_terms(f"{chunk.source} {chunk.heading}"))
            score += 1.25 * len(query_terms & metadata_terms)
            ranked.append(
                KnowledgeSnippet(
                    source=chunk.source,
                    heading=chunk.heading,
                    content=chunk.content,
                    score=score,
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.source, item.heading))
        selected: list[KnowledgeSnippet] = []
        used_characters = 0
        for snippet in ranked:
            size = len(snippet.content) + len(snippet.source) + len(snippet.heading) + 40
            if selected and used_characters + size > self.max_context_characters:
                continue
            selected.append(snippet)
            used_characters += size
            if len(selected) >= self.result_limit or used_characters >= self.max_context_characters:
                break
        return tuple(selected)

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "path": str(self.root_path),
            "available": self.root_path.is_dir() if self.enabled else False,
            "files": self._file_count,
            "chunks": len(self._chunks),
        }

    @staticmethod
    def format_context(snippets: tuple[KnowledgeSnippet, ...]) -> str:
        if not snippets:
            return ""
        sections = []
        for snippet in snippets:
            label = snippet.source
            if snippet.heading:
                label += f" — {snippet.heading}"
            sections.append(f"SOURCE: {label}\n{snippet.content}")
        return "\n\n".join(sections)

    def _build_index(self) -> tuple[list[_IndexedChunk], int]:
        if not self.root_path.is_dir():
            return [], 0
        root = self.root_path.resolve()
        chunks: list[_IndexedChunk] = []
        file_count = 0
        for path in sorted(self.root_path.rglob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                resolved = path.resolve()
                relative = resolved.relative_to(root)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                if resolved.stat().st_size > self.max_file_bytes:
                    continue
                content = resolved.read_text(encoding="utf-8", errors="replace")
                source = relative.as_posix()
            except (OSError, ValueError):
                continue
            file_count += 1
            for heading, text in _markdown_chunks(content, self.chunk_characters):
                terms = Counter(_terms(f"{source} {heading} {text}"))
                if terms:
                    chunks.append(
                        _IndexedChunk(
                            source=source,
                            heading=heading,
                            content=text,
                            terms=terms,
                        )
                    )
        return chunks, file_count


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).casefold()
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _terms(text: str) -> list[str]:
    return [
        term
        for term in _TOKEN_PATTERN.findall(_normalize(text))
        if len(term) >= 2 and term not in _STOP_WORDS and not term.isdecimal()
    ]


def _markdown_chunks(content: str, limit: int) -> list[tuple[str, str]]:
    """Split Markdown on headings/paragraphs while retaining useful local context."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    heading = ""
    pending: list[str] = []
    result: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal pending
        text = "\n\n".join(part.strip() for part in pending if part.strip()).strip()
        pending = []
        while text:
            if len(text) <= limit:
                result.append((heading, text))
                break
            boundary = text.rfind("\n\n", 0, limit + 1)
            if boundary < limit // 3:
                boundary = text.rfind(" ", 0, limit + 1)
            if boundary < limit // 3:
                boundary = limit
            result.append((heading, text[:boundary].strip()))
            text = text[boundary:].strip()

    for block in re.split(r"\n\s*\n", content):
        stripped = block.strip()
        if not stripped:
            continue
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if match and "\n" not in stripped:
            flush()
            heading = match.group(2).strip()
            continue
        if pending and len("\n\n".join(pending)) + len(stripped) + 2 > limit:
            flush()
        pending.append(stripped)
    flush()
    return result

"""Ingestion: load the corpus, split documents into token-bounded chunks, embed
them and write the index (``chunks.jsonl`` + ``embeddings.npy`` + ``manifest.json``).

Chunking is sentence-aligned and token-counted:

* prose is split into sentences and consecutive sentences are packed up to
  ``chunk_tokens``; the next chunk starts with the previous chunk's trailing
  sentences, up to ``chunk_overlap_tokens`` (when the last sentence is longer
  than that, its last words are repeated instead);
* a sentence longer than ``chunk_tokens - chunk_overlap_tokens`` is split at word
  boundaries, so the overlap always fits and is never silently dropped;
* tables (two or more lines containing at least two ``|``) stay whole when they
  fit, otherwise they are split by rows with the header row repeated.

``chunk_id`` is ``"<doc_id>::<position:04d>"``; it depends only on the document
and the chunking settings, so IDs are stable across runs.

    python -m src.ingestion [--config config.yaml] [--provider offline] [--corpus FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass

import numpy as np

from .llm import LLMClient, MissingAPIKeyError
from .retriever import save_index
from .schemas import AppConfig, Document, DocumentChunk, IndexManifest, TokenUsage
from .utils import (
    ConfigError,
    TokenCounter,
    get_logger,
    load_config,
    load_env,
    log_event,
    parse_overrides,
    prompt_hash,
    read_jsonl,
    setup_logging,
    sha256_text,
    utc_now_iso,
)

log = get_logger("ingestion")

_ABBREVIATIONS = frozenset(
    ("e.g i.e etc vs mr mrs ms dr prof inc ltd co corp jr sr st no fig figs al approx dept est u.s").split()
)
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+")
_WORD_SPAN = re.compile(r"\S+")


@dataclass(frozen=True)
class _Unit:
    """An indivisible piece of a chunk: a sentence, a table, or part of either."""

    start: int
    end: int
    text: str  # document[start:end], except table pieces, which repeat the header row
    n_tokens: int
    kind: str  # "sentence" | "words" | "table"


def load_corpus(path: str) -> list[Document]:
    documents = read_jsonl(path, Document)
    seen: set[str] = set()
    for doc in documents:
        if doc.doc_id in seen:
            raise ValueError(f"{path}: duplicate doc_id {doc.doc_id!r}")
        seen.add(doc.doc_id)
    return documents


def corpus_fingerprint(documents: list[Document]) -> str:
    return prompt_hash(sorted((d.doc_id, sha256_text(d.text)) for d in documents))[:16]


# ──────────────────────────────────────────────────────────────── segmentation
def _is_table_line(line: str) -> bool:
    return line.count("|") >= 2


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _blocks(text: str) -> list[tuple[int, int, str]]:
    """Split ``text`` into ``(start, end, kind)`` prose and table blocks."""
    lines = _line_spans(text)
    blocks: list[tuple[int, int, str]] = []
    current: list[tuple[int, int]] = []
    current_kind = ""

    def flush() -> None:
        if current:
            blocks.append((current[0][0], current[-1][1], current_kind))
            current.clear()

    for start, end in lines:
        line = text[start:end]
        if not line.strip():
            flush()
            continue
        kind = "table" if _is_table_line(line) else "prose"
        if current and kind != current_kind:
            flush()
        current_kind = kind
        current.append((start, end))
    flush()
    merged: list[tuple[int, int, str]] = []
    for block in blocks:  # a single "table" line is just prose with pipes
        if block[2] == "table" and "\n" not in text[block[0] : block[1]]:
            block = (block[0], block[1], "prose")
        merged.append(block)
    return merged


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence ``(start, end)`` spans inside ``text[start:end]``, trimmed of whitespace.

    A period after a known abbreviation or a single letter (an initial) does not
    end a sentence; FinQA's detached periods (``"inc ."``) are handled the same way."""
    raw: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        before = text[cursor : match.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
            continue
        raw.append((cursor, match.start() + len(match.group().rstrip())))
        cursor = match.end()
    raw.append((cursor, end))
    spans = []
    for s, e in raw:
        segment = text[s:e]
        stripped = segment.strip()
        if stripped:
            first = s + len(segment) - len(segment.lstrip())
            spans.append((first, first + len(stripped)))
    return spans


def _word_units(text: str, start: int, end: int, counter: TokenCounter, max_tokens: int) -> list[_Unit]:
    """Split ``text[start:end]`` at word boundaries into units of at most ``max_tokens``."""
    units: list[_Unit] = []
    piece_start: int | None = None
    piece_end = start
    used = 0
    for match in _WORD_SPAN.finditer(text, start, end):
        cost = counter.count(match.group())
        if cost > max_tokens:  # one enormous "word" (e.g. a URL): cut it by characters
            if piece_start is not None:
                units.append(_make_unit(text, piece_start, piece_end, counter, "words"))
                piece_start, used = None, 0
            cursor = match.start()
            while cursor < match.end():
                head = counter.truncate(text[cursor : match.end()], max_tokens) or text[cursor]
                units.append(_make_unit(text, cursor, cursor + len(head), counter, "words"))
                cursor += len(head)
            continue
        if piece_start is not None and used + cost > max_tokens:
            units.append(_make_unit(text, piece_start, piece_end, counter, "words"))
            piece_start, used = None, 0
        if piece_start is None:
            piece_start = match.start()
        piece_end, used = match.end(), used + cost
    if piece_start is not None:
        units.append(_make_unit(text, piece_start, piece_end, counter, "words"))
    return units


def _make_unit(text: str, start: int, end: int, counter: TokenCounter, kind: str) -> _Unit:
    piece = text[start:end]
    return _Unit(start, end, piece, counter.count(piece), kind)


def _table_units(text: str, start: int, end: int, counter: TokenCounter, max_tokens: int) -> list[_Unit]:
    whole = _make_unit(text, start, end, counter, "table")
    if whole.n_tokens <= max_tokens:
        return [whole]
    rows = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text[:end]) if m.start() >= start]
    header = text[rows[0][0] : rows[0][1]]
    budget = max_tokens - counter.count(header + "\n")
    if budget < max_tokens // 4:  # header too large to repeat: fall back to word splitting
        return _word_units(text, start, end, counter, max_tokens)
    units: list[_Unit] = []
    group: list[tuple[int, int]] = []

    def emit() -> None:
        body = text[group[0][0] : group[-1][1]]
        piece = body if group[0] == rows[0] else f"{header}\n{body}"
        units.append(_Unit(group[0][0], group[-1][1], piece, counter.count(piece), "table"))
        group.clear()

    for row in rows:
        row_tokens = counter.count(text[row[0] : row[1]])
        if row_tokens > budget:  # a single over-long row
            if group:
                emit()
            units.extend(_word_units(text, row[0], row[1], counter, max_tokens))
            continue
        trial = group + [row]
        body = text[trial[0][0] : trial[-1][1]]
        piece = body if trial[0] == rows[0] else f"{header}\n{body}"
        if group and counter.count(piece) > max_tokens:
            emit()
        group.append(row)
    if group:
        emit()
    return units


def _units(doc: Document, counter: TokenCounter, chunk_tokens: int, overlap_tokens: int) -> list[_Unit]:
    """Sentences, word pieces and tables. Prose units are capped at
    ``chunk_tokens - overlap_tokens`` so the carried-over overlap always fits beside them."""
    prose_limit = chunk_tokens - overlap_tokens
    units: list[_Unit] = []
    for start, end, kind in _blocks(doc.text):
        if kind == "table":
            for unit in _table_units(doc.text, start, end, counter, chunk_tokens):
                if unit.n_tokens <= chunk_tokens:
                    units.append(unit)
                else:  # exact tokenizers are not additive; never exceed the hard limit
                    units.extend(_word_units(doc.text, unit.start, unit.end, counter, chunk_tokens))
            continue
        for s, e in _sentence_spans(doc.text, start, end):
            unit = _make_unit(doc.text, s, e, counter, "sentence")
            if unit.n_tokens <= prose_limit:
                units.append(unit)
            else:
                units.extend(_word_units(doc.text, s, e, counter, prose_limit))
    return units


def _join(text: str, units: list[_Unit]) -> str:
    parts = [units[0].text]
    for prev, cur in zip(units, units[1:], strict=False):
        gap = text[prev.end : cur.start] if prev.end <= cur.start else "\n"
        parts.append(gap if not gap.strip() else "\n")
        parts.append(cur.text)
    return "".join(parts)


def _overlap(units: list[_Unit], text: str, counter: TokenCounter, overlap_tokens: int) -> list[_Unit]:
    """Trailing units of a finished chunk to repeat at the start of the next one."""
    if overlap_tokens <= 0:
        return []
    carried: list[_Unit] = []
    used = 0
    for unit in reversed(units):
        if unit.kind == "table":
            break
        if used + unit.n_tokens <= overlap_tokens:
            carried.insert(0, unit)
            used += unit.n_tokens
            continue
        if not carried:  # last sentence longer than the overlap: carry its final words
            words = list(_WORD_SPAN.finditer(text, unit.start, unit.end))
            budget, first = overlap_tokens, None
            for match in reversed(words):
                cost = counter.count(match.group())
                if cost > budget:
                    break
                budget -= cost
                first = match.start()
            if first is not None:
                carried = [_make_unit(text, first, unit.end, counter, "words")]
        break
    return carried


def chunk_document(doc: Document, counter: TokenCounter, chunk_tokens: int, overlap_tokens: int) -> list[DocumentChunk]:
    """Token-bounded, sentence-aligned chunks with overlap (see the module docstring)."""
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []  # always ends with at least one unit not yet emitted
    for unit in _units(doc, counter, chunk_tokens, overlap_tokens):
        if current and counter.count(_join(doc.text, [*current, unit])) > chunk_tokens:
            groups.append(current)
            current = _overlap(current, doc.text, counter, overlap_tokens)
            while current and counter.count(_join(doc.text, [*current, unit])) > chunk_tokens:
                current = current[1:]
        current = [*current, unit]
    if current:
        groups.append(current)

    metadata = dict(doc.metadata)
    if doc.title:
        metadata.setdefault("title", doc.title)
    chunks = []
    for position, group in enumerate(groups):
        body = _join(doc.text, group)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{doc.doc_id}::{position:04d}",
                doc_id=doc.doc_id,
                position=position,
                text=body,
                n_tokens=counter.count(body),
                start_char=group[0].start,
                end_char=group[-1].end,
                content_hash=sha256_text(body)[:16],
                metadata=metadata,
            )
        )
    return chunks


def chunk_corpus(documents: list[Document], cfg: AppConfig, counter: TokenCounter) -> list[DocumentChunk]:
    return [
        chunk
        for doc in documents
        for chunk in chunk_document(doc, counter, cfg.ingestion.chunk_tokens, cfg.ingestion.chunk_overlap_tokens)
    ]


# ──────────────────────────────────────────────────────────────── index build
async def build_index(cfg: AppConfig, llm: LLMClient, documents: list[Document] | None = None) -> IndexManifest:
    """Chunk and embed the corpus and write the index to ``ingestion.index_dir``."""
    docs = documents if documents is not None else load_corpus(cfg.ingestion.corpus_path)
    counter = TokenCounter(cfg.tokens.encoding)
    chunks = chunk_corpus(docs, cfg, counter)
    if not chunks:
        raise ValueError("the corpus produced no chunks (are all documents empty?)")
    size = cfg.ingestion.embed_batch_size
    batches = [chunks[i : i + size] for i in range(0, len(chunks), size)]
    results = await asyncio.gather(*(llm.embed([c.text for c in b], task="embed_documents") for b in batches))
    embeddings = np.concatenate([matrix for matrix, _ in results], axis=0)
    usage = TokenUsage.total([u for _, u in results])
    manifest = IndexManifest(
        embedding_provider=cfg.llm.provider,
        embedding_model=cfg.models.embedding,
        dimension=int(embeddings.shape[1]),
        normalized=True,
        token_encoding=counter.name,
        chunk_tokens=cfg.ingestion.chunk_tokens,
        chunk_overlap_tokens=cfg.ingestion.chunk_overlap_tokens,
        n_documents=len(docs),
        n_chunks=len(chunks),
        corpus_hash=corpus_fingerprint(docs),
        created_at=utc_now_iso(),
    )
    save_index(cfg.ingestion.index_dir, chunks, embeddings, manifest)
    log_event(
        log,
        "index written",
        index_dir=cfg.ingestion.index_dir,
        documents=len(docs),
        chunks=len(chunks),
        embedding_tokens=usage.embedding_tokens,
        cost_usd=round(usage.cost_usd, 6),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk, embed and index the corpus.")
    parser.add_argument("--config", default=None, help="YAML config (default: ./config.yaml)")
    parser.add_argument("--corpus", help="corpus JSONL (overrides ingestion.corpus_path)")
    parser.add_argument("--index-dir", help="output directory (overrides ingestion.index_dir)")
    parser.add_argument("--provider", choices=["openai", "offline"], help="overrides llm.provider")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="any config override")
    return parser


async def _run(cfg: AppConfig) -> IndexManifest:
    llm = LLMClient.from_config(cfg)
    try:
        return await build_index(cfg, llm)
    finally:
        await llm.aclose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env()
    try:
        overrides = parse_overrides(args.set)
        for key, value in (
            ("ingestion.corpus_path", args.corpus),
            ("ingestion.index_dir", args.index_dir),
            ("llm.provider", args.provider),
        ):
            if value is not None:
                overrides[key] = value
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.run.log_level, cfg.run.log_format)
    try:
        manifest = asyncio.run(_run(cfg))
    except (MissingAPIKeyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"indexed {manifest.n_documents} documents → {manifest.n_chunks} chunks "
        f"({manifest.embedding_provider}:{manifest.embedding_model}, dim {manifest.dimension}, "
        f"tokens: {manifest.token_encoding}) in {cfg.ingestion.index_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

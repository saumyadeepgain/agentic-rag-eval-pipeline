"""Shared fixtures. Nothing in the test suite touches the network or needs an API key:
LLM calls go to ``ScriptedChat`` (queued replies per task) or ``OfflineBackend``, and
the retriever is either ``FakeRetriever`` or a real index built with hashed embeddings."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import build_index  # noqa: E402
from src.llm import ChatRequest, ChatResponse, LLMClient, OfflineBackend  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.schemas import (  # noqa: E402
    AppConfig,
    Document,
    DocumentChunk,
    EvalSample,
    IndexManifest,
    RetrievalOutcome,
    RetrievalResult,
)
from src.utils import load_config, sha256_text  # noqa: E402

CONFIG = ROOT / "config.yaml"


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def make_cfg(tmp_path: Path, mode: str | None = "agentic", overrides: dict[str, Any] | None = None) -> AppConfig:
    """The repository's config.yaml, offline, with every path inside ``tmp_path``."""
    base: dict[str, Any] = {
        "llm.provider": "offline",
        "tokens.encoding": "approx",
        "cache.enabled": False,
        "cache.path": str(tmp_path / "cache.sqlite"),
        "ingestion.index_dir": str(tmp_path / "index"),
        "run.results_dir": str(tmp_path / "results"),
        "llm.backoff_initial_s": 0.0,
        "llm.backoff_max_s": 0.0,
        "evaluation.bootstrap_samples": 200,
    }
    base.update(overrides or {})
    return load_config(CONFIG, mode=mode, overrides=base)


Reply = Any  # dict (sent as JSON) | str (raw text, e.g. malformed) | BaseException (raised)


class ScriptedChat:
    """Chat backend with queued replies per task. When a task's queue is empty the
    ``defaults[task](request)`` callable answers; a task with neither fails the test."""

    name = "scripted"

    def __init__(
        self,
        replies: dict[str, list[Reply]] | None = None,
        defaults: dict[str, Callable[[ChatRequest], Reply]] | None = None,
    ) -> None:
        self.queues = {task: list(items) for task, items in (replies or {}).items()}
        self.defaults = defaults or {}
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        queue = self.queues.get(request.task)
        if queue:
            reply = queue.pop(0)
        elif request.task in self.defaults:
            reply = self.defaults[request.task](request)
        else:
            raise AssertionError(f"unexpected LLM call for task {request.task!r}")
        if isinstance(reply, BaseException):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return ChatResponse(text=text, prompt_tokens=100, completion_tokens=20)

    def calls(self, task: str) -> list[ChatRequest]:
        return [r for r in self.requests if r.task == task]


def chunk(chunk_id: str, text: str, **metadata: Any) -> DocumentChunk:
    doc_id = chunk_id.split("::")[0]
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        position=int(chunk_id.split("::")[1]) if "::" in chunk_id else 0,
        text=text,
        n_tokens=len(text.split()),
        start_char=0,
        end_char=len(text),
        content_hash=sha256_text(text)[:16],
        metadata=metadata,
    )


def result(chunk_id: str, text: str, rank: int = 1, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=chunk(chunk_id, text), rank=rank, score=score)


class FakeRetriever:
    """Returns ``rounds[i]`` on the i-th call (the last round repeats); can raise instead."""

    def __init__(self, rounds: list[list[RetrievalResult]], error: BaseException | None = None) -> None:
        self.rounds = rounds
        self.error = error
        self.queries: list[str] = []
        self.manifest = IndexManifest(
            embedding_provider="offline",
            embedding_model="fake",
            dimension=1,
            token_encoding="approx",
            chunk_tokens=200,
            chunk_overlap_tokens=40,
            n_documents=0,
            n_chunks=0,
            corpus_hash="fake",
            created_at="2026-01-01T00:00:00+00:00",
        )

    async def retrieve(self, query: str, *, top_k: int | None = None, filters: Any = None) -> RetrievalOutcome:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        results = self.rounds[min(len(self.queries), len(self.rounds)) - 1] if self.rounds else []
        return RetrievalOutcome(results=list(results), retrieval_ms=1.0, rerank_ms=0.5)


def client(cfg: AppConfig, chat: ScriptedChat | None = None) -> LLMClient:
    offline = OfflineBackend()
    return LLMClient(cfg, chat_backend=chat or offline, embed_backend=offline)


def sample(question: str = "What was ACME revenue in 2020?", **kwargs: Any) -> EvalSample:
    return EvalSample(query_id=kwargs.pop("query_id", "q1"), question=question, **kwargs)


CORPUS = [
    Document(
        doc_id="acme-1",
        text="ACME revenue in 2020 was 5 million dollars. Revenue grew 10 percent.",
        metadata={"domain": "finance"},
    ),
    Document(doc_id="acme-2", text="The ACME CEO is Jane Smith. She joined in 2018.", metadata={"domain": "finance"}),
    Document(
        doc_id="med-1", text="Aspirin reduces fever. The usual adult dose is 500 mg.", metadata={"domain": "medical"}
    ),
    Document(
        doc_id="med-2",
        text="Hydration matters for older adults. Most drink six glasses a day.",
        metadata={"domain": "medical"},
    ),
]


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return make_cfg(tmp_path)


@pytest.fixture
def built_index(tmp_path: Path) -> AppConfig:
    """A real index of ``CORPUS`` (offline hashed embeddings); returns its config (hybrid mode)."""
    config = make_cfg(tmp_path, mode="hybrid_rerank")
    llm = client(config)
    run(build_index(config, llm, CORPUS))
    return config


@pytest.fixture
def retriever(built_index: AppConfig) -> Retriever:
    return Retriever.load(built_index, client(built_index))

"""Async LLM and embedding client shared by every call site.

Router, grader, rewriter, generator, judges and embeddings all go through
``LLMClient``, so every call gets the same guarantees:

* temperature 0 and JSON output validated against a Pydantic schema, with one
  repair round-trip when the output does not parse;
* a SQLite response cache keyed by (provider, model, prompt hash), so re-running
  an evaluation does not pay for the same calls twice;
* exponential backoff with jitter (tenacity) on rate limits, timeouts and 5xx,
  inside an ``asyncio.Semaphore`` that caps concurrent API calls;
* prompt/completion tokens, latency and list-price cost for every call.

Backends:

* ``OpenAIBackend``: any OpenAI-compatible endpoint (OpenAI, Gemini's
  compatibility endpoint, vLLM, Ollama, ...), selected by ``llm.base_url``.
* ``OfflineBackend``: deterministic word-overlap heuristics and hashed
  embeddings with no network access, for smoke tests and CI. Its answers and
  scores say nothing about model quality.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from .schemas import AppConfig, LLMUsage
from .utils import ResponseCache, get_logger, log_event, prompt_hash

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
log = get_logger("llm")


# ──────────────────────────────────────────────────────────────── errors
class LLMError(RuntimeError):
    """A call failed (after retries, for transient errors)."""

    usage: LLMUsage | None = None


class TransientLLMError(LLMError):
    """Retryable failure: rate limit, timeout or server error."""


class MissingAPIKeyError(LLMError):
    """The environment variable named by ``llm.api_key_env`` is not set."""


class StructuredOutputError(LLMError):
    """The model's reply was not valid JSON for the requested schema."""

    def __init__(self, task: str, raw: str, error: str, usage: LLMUsage | None = None) -> None:
        super().__init__(f"{task}: reply is not valid JSON for the schema ({error}); got {raw[:160]!r}")
        self.task, self.raw, self.error, self.usage = task, raw, error, usage


def is_transient(exc: BaseException) -> bool:
    """Errors worth retrying: rate limits, timeouts, dropped connections, 5xx."""
    if isinstance(exc, (TransientLLMError, asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    try:
        import openai
    except ImportError:
        return False
    return isinstance(
        exc,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError),
    )


# ──────────────────────────────────────────────────────────────── backends
@dataclass
class ChatRequest:
    model: str
    messages: list[dict[str, str]]
    task: str
    schema_name: str
    json_schema: dict[str, Any]
    max_tokens: int
    temperature: float = 0.0
    seed: int | None = None
    response_format: str = "json_object"
    payload: dict[str, Any] = field(default_factory=dict)  # structured inputs (offline backend, mocks)


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatBackend(Protocol):
    name: str

    async def chat(self, request: ChatRequest) -> ChatResponse: ...


class EmbeddingBackend(Protocol):
    name: str

    async def embed(self, model: str, texts: list[str]) -> tuple[np.ndarray, int]: ...


class OpenAIBackend:
    """Chat completions and embeddings on any OpenAI-compatible API.

    The SDK's own retries are disabled (``max_retries=0``) so that tenacity in
    ``LLMClient`` is the only retry layer."""

    name = "openai"

    def __init__(self, cfg: AppConfig, client: Any = None) -> None:
        if client is None:
            api_key = os.environ.get(cfg.llm.api_key_env, "").strip()
            if not api_key:
                raise MissingAPIKeyError(
                    f"environment variable {cfg.llm.api_key_env} is not set (add it to .env), "
                    "or run with --provider offline for a keyless smoke test"
                )
            import openai

            client = openai.AsyncOpenAI(
                api_key=api_key, base_url=cfg.llm.base_url, timeout=cfg.llm.timeout_s, max_retries=0
            )
        self._client = client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.response_format == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "schema": request.json_schema},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message if response.choices else None
        usage = response.usage
        return ChatResponse(
            text=(message.content if message is not None else None) or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    async def embed(self, model: str, texts: list[str]) -> tuple[np.ndarray, int]:
        response = await self._client.embeddings.create(model=model, input=texts)
        rows = sorted(response.data, key=lambda item: item.index)
        vectors = np.asarray([row.embedding for row in rows], dtype=np.float32)
        return vectors, int(getattr(response.usage, "prompt_tokens", 0) or 0)

    async def aclose(self) -> None:
        await self._client.close()


_STOPWORDS = frozenset(
    (
        "a an the of to in on for and or is are was were be been being by with as at from that this "
        "these those it its what which who whom whose when where why how do does did can could should "
        "would will may might there their than then into about over under between during per vs not "
        "no any all each other such only own same so too very just also"
    ).split()
)
_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def content_terms(text: str) -> list[str]:
    """Lower-cased words without stop words (shared by the offline backend and the lexical reranker)."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def _approx_tokens(text: str) -> int:
    return max(1, len(_WORD.findall(text)))


class OfflineBackend:
    """Deterministic stand-ins for the chat model and the embedding model.

    Decisions come from word overlap between the question and the text, so the
    whole pipeline (routing, grading, rewriting, generation, judging) runs with
    no network and no API key. Use it for smoke tests and CI only."""

    name = "offline"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    async def chat(self, request: ChatRequest) -> ChatResponse:
        handler = getattr(self, f"_task_{request.task}", None)
        if handler is None:
            raise LLMError(f"offline backend has no handler for task {request.task!r}")
        text = json.dumps(handler(request.payload), ensure_ascii=False)
        prompt = " ".join(m.get("content", "") for m in request.messages)
        return ChatResponse(text=text, prompt_tokens=_approx_tokens(prompt), completion_tokens=_approx_tokens(text))

    async def embed(self, model: str, texts: list[str]) -> tuple[np.ndarray, int]:
        vectors = np.stack([self._hash_vector(t) for t in texts]) if texts else np.zeros((0, self.dimension))
        return vectors.astype(np.float32), sum(_approx_tokens(t) for t in texts)

    def _hash_vector(self, text: str) -> np.ndarray:
        terms = content_terms(text)
        features = Counter(terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:], strict=False)])
        vector = np.zeros(self.dimension, dtype=np.float64)
        for feature, count in features.items():
            digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
            sign = 1.0 if digest >> 63 else -1.0
            vector[digest % self.dimension] += sign * (1.0 + math.log(count))
        return vector

    @staticmethod
    def _overlap(query: str, text: str) -> tuple[int, float]:
        wanted = set(content_terms(query))
        found = wanted & set(content_terms(text))
        return len(found), (len(found) / len(wanted) if wanted else 0.0)

    def _task_route(self, p: dict[str, Any]) -> dict[str, Any]:
        has_terms = bool(content_terms(p["question"]))
        return {
            "reasoning": "offline heuristic: retrieve whenever the question has content words",
            "action": "retrieve" if has_terms else "refuse",
            "search_query": p["question"],
        }

    def _task_grade(self, p: dict[str, Any]) -> dict[str, Any]:
        hits, ratio = self._overlap(p["question"], p["chunk_text"])
        return {
            "reasoning": f"offline heuristic: {hits} question terms found ({ratio:.0%})",
            "relevant": hits >= 2 and ratio >= 0.3,
        }

    def _task_rewrite(self, p: dict[str, Any]) -> dict[str, Any]:
        keywords = " ".join(dict.fromkeys(content_terms(p["question"])))
        query = keywords if keywords and keywords not in p.get("tried", []) else p["previous_query"]
        return {"reasoning": "offline heuristic: keep the question's content words", "query": query}

    def _task_generate(self, p: dict[str, Any]) -> dict[str, Any]:
        best: tuple[int, str, str] = (0, "", "")
        for doc in p["documents"]:
            for sentence in _SENTENCE_BREAK.split(doc["text"]):
                hits, _ = self._overlap(p["question"], sentence)
                if hits > best[0]:
                    best = (hits, sentence.strip(), doc["chunk_id"])
        if best[0] < 2:
            return {"answer": "", "citations": [], "abstained": True}
        return {"answer": best[1][:400], "citations": [best[2]], "abstained": False}

    def _task_judge_faithfulness(self, p: dict[str, Any]) -> dict[str, Any]:
        claims = []
        for sentence in (s.strip() for s in _SENTENCE_BREAK.split(p["answer"])):
            if not content_terms(sentence):
                continue
            support = max(
                ((self._overlap(sentence, c["text"])[1], c["chunk_id"]) for c in p["contexts"]),
                default=(0.0, None),
            )
            claims.append({"claim": sentence, "supported": support[0] >= 0.6, "evidence": support[1]})
        return {
            "reasoning": "offline heuristic: a claim is supported when 60% of its words occur in one chunk",
            "claims": claims,
        }

    def _task_judge_relevancy(self, p: dict[str, Any]) -> dict[str, Any]:
        _, ratio = self._overlap(p["question"], p["answer"])
        return {
            "reasoning": f"offline heuristic: answer covers {ratio:.0%} of the question's words",
            "score": 1 + round(4 * ratio),
        }

    def _task_judge_context(self, p: dict[str, Any]) -> dict[str, Any]:
        hits, ratio = self._overlap(f"{p['question']} {p.get('reference') or ''}", p["chunk_text"])
        return {
            "reasoning": f"offline heuristic: {hits} shared terms ({ratio:.0%})",
            "useful": hits >= 2 and ratio >= 0.2,
        }


# ──────────────────────────────────────────────────────────────── parsing helpers
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_output(text: str, schema: type[T]) -> T:
    """Validate a model reply against ``schema``; tolerates code fences and text around the object."""
    cleaned = (text or "").strip()
    fenced = _FENCE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1)
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the reply")
        cleaned = cleaned[start : end + 1]
    return schema.model_validate_json(cleaned)


def _describe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "reply"
        return f"{where}: {first.get('msg', 'invalid')}"
    return str(exc)


def _with_schema_instruction(messages: list[dict[str, str]], json_schema: dict[str, Any]) -> list[dict[str, str]]:
    instruction = (
        "Respond with one JSON object and nothing else. It must match this JSON schema; "
        "write the fields in the order given (reasoning first, then the verdict):\n"
        + json.dumps(json_schema, ensure_ascii=False)
    )
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{instruction}"
    else:
        out.insert(0, {"role": "system", "content": instruction})
    return out


def _encode_vector(vector: np.ndarray) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def _decode_vector(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32).copy()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows (zero rows stay zero), so a dot product is cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


# ──────────────────────────────────────────────────────────────── client
class LLMClient:
    """See the module docstring. Create one per run with ``LLMClient.from_config``."""

    def __init__(
        self,
        cfg: AppConfig,
        chat_backend: ChatBackend | None = None,
        embed_backend: EmbeddingBackend | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.cfg = cfg
        if chat_backend is None or embed_backend is None:
            default: Any = OfflineBackend() if cfg.llm.provider == "offline" else OpenAIBackend(cfg)
            chat_backend = chat_backend or default
            embed_backend = embed_backend or default
        self.chat_backend: ChatBackend = chat_backend
        self.embed_backend: EmbeddingBackend = embed_backend
        self.cache = cache
        self.namespace = f"{cfg.llm.provider}|{cfg.llm.base_url or 'default'}"
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self._unpriced: set[str] = set()

    @classmethod
    def from_config(cls, cfg: AppConfig) -> LLMClient:
        return cls(cfg, cache=ResponseCache(cfg.cache.path) if cfg.cache.enabled else None)

    @property
    def provider(self) -> str:
        return self.cfg.llm.provider

    def model_for(self, role: str) -> str:
        models = self.cfg.models
        roles = {"generator": models.generator, "agent": models.agent, "judge": models.judge}
        if role not in roles:
            raise ValueError(f"unknown model role {role!r}")
        return roles[role]

    @staticmethod
    def render_messages(messages: list[dict[str, str]], schema: type[BaseModel]) -> list[dict[str, str]]:
        """The messages exactly as sent for ``schema`` (used for prompt-token budgets)."""
        return _with_schema_instruction(messages, schema.model_json_schema())

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """List-price cost in USD; ``None`` when the model has no price in the config."""
        if self.provider == "offline":
            return 0.0
        price = self.cfg.pricing.get(model)
        if price is None:
            if model not in self._unpriced:
                self._unpriced.add(model)
                log_event(log, "no price configured; cost reported as unknown", logging.WARNING, model=model)
            return None
        return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000

    def _semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in self._semaphores:
            self._semaphores = {loop_id: asyncio.Semaphore(self.cfg.llm.max_concurrency)}
        return self._semaphores[loop_id]

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log_event(
            log,
            "transient API error, backing off",
            logging.WARNING,
            attempt=state.attempt_number,
            sleep_s=round(state.next_action.sleep, 2) if state.next_action else None,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )

    async def _call_with_retries(self, call: Callable[[], Awaitable[R]]) -> tuple[R, int]:
        attempts = 0
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.cfg.llm.max_retries),
            # initial * 2^(attempt-1), capped, plus up to `initial` seconds of jitter
            wait=wait_exponential(multiplier=self.cfg.llm.backoff_initial_s, max=self.cfg.llm.backoff_max_s)
            + wait_random(0, self.cfg.llm.backoff_initial_s),
            retry=retry_if_exception(is_transient),
            before_sleep=self._log_retry,
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    attempts += 1
                    async with self._semaphore():
                        result = await call()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__} after {attempts} attempt(s): {exc}") from exc
        return result, attempts

    async def structured(
        self,
        *,
        task: str,
        role: str,
        messages: list[dict[str, str]],
        schema: type[T],
        payload: dict[str, Any],
        max_tokens: int,
    ) -> tuple[T, LLMUsage]:
        """One schema-validated call. Raises ``StructuredOutputError`` or ``LLMError``
        (both carry ``.usage``); callers decide the fallback."""
        model = self.model_for(role)
        json_schema = schema.model_json_schema()
        conversation = _with_schema_instruction(messages, json_schema)
        response_format = self.cfg.llm.response_format
        key = prompt_hash(
            {
                "messages": conversation,
                "schema": json_schema,
                "max_tokens": max_tokens,
                "format": response_format,
                "seed": self.cfg.llm.seed,
                "temperature": 0.0,
            }
        )
        cache_model = f"{self.namespace}|{model}"
        usage = LLMUsage(task=task, model=model)
        started = time.perf_counter()

        if self.cache is not None:
            hit = self.cache.get(cache_model, key)
            if hit is not None:
                try:
                    parsed = parse_json_output(hit["text"], schema)
                except (ValidationError, ValueError):
                    parsed = None  # entry written for an older schema; fall through to a fresh call
                if parsed is not None:
                    usage.prompt_tokens = int(hit.get("prompt_tokens", 0))
                    usage.completion_tokens = int(hit.get("completion_tokens", 0))
                    usage.cached = True
                    usage.cost_usd = self.cost(model, usage.prompt_tokens, usage.completion_tokens)
                    usage.latency_ms = (time.perf_counter() - started) * 1000
                    return parsed, usage

        def finish() -> LLMUsage:
            usage.latency_ms = (time.perf_counter() - started) * 1000
            usage.cost_usd = self.cost(model, usage.prompt_tokens, usage.completion_tokens)
            return usage

        for parse_try in range(self.cfg.llm.max_parse_retries + 1):
            request = ChatRequest(
                model=model,
                messages=conversation,
                task=task,
                schema_name=schema.__name__,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=self.cfg.llm.seed,
                response_format=response_format,
                payload=payload,
            )
            try:
                response, attempts = await self._call_with_retries(
                    lambda request=request: self.chat_backend.chat(request)
                )
            except LLMError as exc:
                exc.usage = finish()
                raise
            usage.attempts += attempts
            usage.prompt_tokens += response.prompt_tokens
            usage.completion_tokens += response.completion_tokens
            try:
                parsed = parse_json_output(response.text, schema)
            except (ValidationError, ValueError) as exc:
                usage.parse_retries = parse_try + 1
                problem = _describe_error(exc)
                if parse_try >= self.cfg.llm.max_parse_retries:
                    raise StructuredOutputError(task, response.text, problem, finish()) from exc
                log_event(
                    log, "reply did not match the schema, asking again", logging.WARNING, task=task, error=problem
                )
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": response.text or "(empty reply)"},
                    {
                        "role": "user",
                        "content": f"That reply was not a valid JSON object for the schema ({problem}). "
                        "Reply with only the corrected JSON object.",
                    },
                ]
                continue
            if self.cache is not None:
                self.cache.put(
                    cache_model,
                    key,
                    {
                        "text": response.text,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    },
                )
            return parsed, finish()
        raise AssertionError("unreachable: the loop returns or raises")

    async def embed(self, texts: list[str], *, task: str = "embed") -> tuple[np.ndarray, LLMUsage]:
        """L2-normalised embeddings (float32, one row per text), cached per text."""
        model = self.cfg.models.embedding
        usage = LLMUsage(task=task, model=model)
        started = time.perf_counter()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), usage
        cache_model = f"{self.namespace}|{model}"
        keys = [prompt_hash({"embed": text}) for text in texts]
        vectors: list[np.ndarray | None] = [None] * len(texts)
        cached_tokens = 0
        if self.cache is not None:
            for i, key in enumerate(keys):
                hit = self.cache.get(cache_model, key)
                if hit is not None:
                    vectors[i] = _decode_vector(hit["vector"])
                    cached_tokens += int(hit.get("tokens", 0))
        missing = [i for i, vector in enumerate(vectors) if vector is None]
        fresh_tokens = 0
        if missing:
            batch = [texts[i] for i in missing]
            try:
                (matrix, fresh_tokens), attempts = await self._call_with_retries(
                    lambda: self.embed_backend.embed(model, batch)
                )
            except LLMError as exc:
                usage.latency_ms = (time.perf_counter() - started) * 1000
                exc.usage = usage
                raise
            if matrix.shape[0] != len(batch):
                raise LLMError(f"embedding backend returned {matrix.shape[0]} vectors for {len(batch)} texts")
            usage.attempts = attempts
            total_chars = sum(len(t) for t in batch) or 1
            for row, i in enumerate(missing):
                vectors[i] = matrix[row]
                if self.cache is not None:
                    share = round(fresh_tokens * len(texts[i]) / total_chars)
                    self.cache.put(cache_model, keys[i], {"vector": _encode_vector(matrix[row]), "tokens": share})
        stacked = l2_normalize(np.stack([v for v in vectors if v is not None]))
        usage.prompt_tokens = fresh_tokens + cached_tokens
        usage.cached = not missing
        usage.cost_usd = self.cost(model, usage.prompt_tokens, 0)
        usage.latency_ms = (time.perf_counter() - started) * 1000
        return stacked, usage

    async def aclose(self) -> None:
        for backend in {id(self.chat_backend): self.chat_backend, id(self.embed_backend): self.embed_backend}.values():
            closer = getattr(backend, "aclose", None)
            if closer is not None:
                await closer()
        if self.cache is not None:
            self.cache.close()

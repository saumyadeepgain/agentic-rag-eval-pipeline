"""Shared helpers: configuration (YAML + mode presets + CLI overrides), logging,
token counting, the SQLite response cache, hashing, git provenance and JSON I/O."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .schemas import AppConfig

T = TypeVar("T", bound=BaseModel)
LOGGER_NAME = "rag"
DEFAULT_CONFIG = "config.yaml"


# ──────────────────────────────────────────────────────────────── configuration
class ConfigError(ValueError):
    """The configuration file or an override is invalid."""


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """``["retrieval.top_k=5", "agent.enabled=false"]`` → ``{"retrieval.top_k": 5, ...}``.

    Values are parsed as YAML, so ``5``, ``0.5``, ``true``, ``null`` and ``[1, 3]``
    become typed values and anything else stays a string."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override must look like section.key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(raw) if raw.strip() else ""
    return out


def set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    """Set ``data["a"]["b"] = value`` for ``key == "a.b"``, creating sections as needed."""
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set {key!r}: {part!r} is not a section")
        node = child
    node[parts[-1]] = value


def load_config(
    path: str | Path | None = None,
    *,
    mode: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load ``config.yaml``, apply the mode preset, then explicit overrides (which win).

    ``path=None`` uses ``./config.yaml`` when it exists and the built-in defaults
    otherwise. Unknown keys are rejected by the schema."""
    overrides = dict(overrides or {})
    if path is None and Path(DEFAULT_CONFIG).is_file():
        path = DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if path is not None:
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a mapping at the top level")

    chosen = mode or overrides.pop("run.mode", None) or data.get("run", {}).get("mode")
    presets = data.get("modes") or {}
    if chosen is not None:
        if presets and chosen not in presets:
            raise ConfigError(f"unknown mode {chosen!r}; choose from {sorted(presets)}")
        for key, value in (presets.get(chosen) or {}).items():
            set_dotted(data, key, value)
        set_dotted(data, "run.mode", chosen)
    for key, value in overrides.items():
        set_dotted(data, key, value)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc


def config_to_yaml(cfg: AppConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def config_hash(cfg: AppConfig) -> str:
    return prompt_hash(cfg.model_dump(mode="json"))[:16]


def load_env() -> None:
    """Load ``.env`` into the environment without overriding variables already set."""
    from dotenv import load_dotenv

    load_dotenv(override=False)


# ──────────────────────────────────────────────────────────────────── logging
class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", None) or {}
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        extra = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
        line = f"{stamp} {record.levelname:<7} {record.name}: {record.getMessage()}"
        line = f"{line} {extra}" if extra else line
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **(getattr(record, "fields", None) or {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def setup_logging(level: str = "INFO", fmt: str = "text", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the ``rag`` logger: stderr plus an optional file (always JSON lines)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_rag_handler", False):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(level.upper())
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(_JsonFormatter() if fmt == "json" else _TextFormatter())
    stream._rag_handler = True  # type: ignore[attr-defined]
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(_JsonFormatter())
        file_handler._rag_handler = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log ``event`` with structured key/value fields (rendered as JSON in log files)."""
    logger.log(level, event, extra={"fields": fields})


# ──────────────────────────────────────────────────────────────── token counting
_APPROX_PIECES = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_log = get_logger("utils")


class TokenCounter:
    """Counts tokens with tiktoken when the encoding is available.

    ``encoding="approx"`` (or an encoding that cannot be loaded, e.g. offline)
    uses a deterministic estimate: one token per punctuation mark and per word,
    plus one per 8 characters of long words. The estimate is additive across
    whitespace boundaries, which the chunker relies on."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        self.requested = encoding
        self.name = "approx"
        self._enc: Any = None
        if encoding != "approx":
            try:
                import tiktoken

                self._enc = tiktoken.get_encoding(encoding)
                self.name = encoding
            except Exception as exc:  # no tiktoken, unknown encoding or no download access
                log_event(
                    _log,
                    "tiktoken encoding unavailable, using approximate token counts",
                    logging.WARNING,
                    encoding=encoding,
                    reason=f"{type(exc).__name__}: {exc}"[:200],
                )

    @property
    def exact(self) -> bool:
        return self._enc is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        return sum(1 + len(piece) // 8 for piece in _APPROX_PIECES.findall(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Longest prefix of ``text`` with at most ``max_tokens`` tokens."""
        if max_tokens <= 0:
            return ""
        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            return text if len(ids) <= max_tokens else self._enc.decode(ids[:max_tokens])
        used, end = 0, 0
        for match in _APPROX_PIECES.finditer(text):
            cost = 1 + len(match.group()) // 8
            if used + cost > max_tokens:
                break
            used += cost
            end = match.end()
        return text if end == 0 and not text.strip() else text[:end]


# ──────────────────────────────────────────────────────────────── response cache
class ResponseCache:
    """SQLite cache of LLM responses keyed by ``(model, prompt_hash)``.

    Safe to share between coroutines and threads; values are JSON objects."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        # WAL is an optimisation; some filesystems refuse it and the default journal works
        with contextlib.suppress(sqlite3.DatabaseError):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " model TEXT NOT NULL, prompt_hash TEXT NOT NULL, response TEXT NOT NULL,"
            " created_at TEXT NOT NULL, PRIMARY KEY (model, prompt_hash))"
        )
        self._conn.commit()

    def get(self, model: str, prompt_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE model = ? AND prompt_hash = ?",
                (model, prompt_hash),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, prompt_hash: str, value: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(value), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                (model, prompt_hash, payload, utc_now_iso()),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ──────────────────────────────────────────────────────────────── hashing
def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_hash(obj: Any) -> str:
    return sha256_text(stable_json(obj))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ──────────────────────────────────────────────────────────────── provenance
def git_info(cwd: str | Path | None = None) -> tuple[str | None, bool | None]:
    """``(commit, dirty)`` of the repository at ``cwd``; ``(None, None)`` outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class Stopwatch:
    """``with Stopwatch() as sw: ...`` then ``sw.ms``."""

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0


# ──────────────────────────────────────────────────────────────── JSON I/O
def _to_jsonable(obj: Any) -> Any:
    return obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: str | Path, obj: Any) -> None:
    text = json.dumps(_to_jsonable(obj), indent=2, ensure_ascii=False, default=str)
    _atomic_write(Path(path), text + "\n")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    lines = (json.dumps(_to_jsonable(r), ensure_ascii=False, default=str) for r in records)
    _atomic_write(Path(path), "".join(f"{line}\n" for line in lines))


def read_jsonl(path: str | Path, model: type[T]) -> list[T]:
    """Parse and validate a JSONL file, reporting the line number of a bad record."""
    out: list[T] = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                out.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path}:{number}: invalid {model.__name__}: {exc}") from exc
    return out

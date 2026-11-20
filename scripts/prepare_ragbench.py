"""Convert the RAGBench samples in ``data/ragbench_samples/`` into a pooled corpus
and an evaluation set with gold document IDs.

Each RAGBench row pairs a question with the documents retrieved for it. Here the
documents of the selected domains are pooled into one corpus, so every other
question's documents act as distractors, and each question's own documents
become its ``gold_doc_ids`` (an approximation: a few of them are off-topic).

RAGBench has no gold answer column. Its ``response`` column, kept here as
``reference_answer`` with ``reference_source="ragbench_response"``, is an
LLM-generated answer that RAGBench itself annotates for hallucinations.

FinQA tables arrive as JSON-encoded lists of rows; they are rendered as pipe
tables so the chunker keeps each table (and its header) together.

    python scripts/prepare_ragbench.py --domains finance biomedical legal \
        --corpus-out data/ragbench_corpus.jsonl --eval-out data/ragbench_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DOMAINS = {"finance": ("fin", "finqa"), "biomedical": ("bio", "pubmedqa"), "legal": ("legal", "cuad")}


def render_table(text: str) -> str | None:
    """``'[["", "2010"], ["a", "1"]]'`` → a pipe table; ``None`` if ``text`` is not a table."""
    try:
        rows = json.loads(text)
    except ValueError:
        return None
    if not (isinstance(rows, list) and len(rows) >= 2 and all(isinstance(r, list) and r for r in rows)):
        return None
    return "\n".join("| " + " | ".join(str(cell).strip() or "-" for cell in row) + " |" for row in rows)


def load_domain(samples_dir: Path, domain: str) -> list[dict[str, Any]]:
    files = sorted(samples_dir.glob(f"{domain}_*.json"))
    if not files:
        raise FileNotFoundError(f"no {domain}_*.json in {samples_dir}")
    return json.loads(files[0].read_text(encoding="utf-8"))


def convert(samples_dir: Path, domains: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for domain in domains:
        prefix, source = DOMAINS[domain]
        for q, row in enumerate(load_domain(samples_dir, domain)):
            gold = []
            for d, text in enumerate(row["documents"]):
                doc_id = f"{prefix}-{q:02d}-{d}"
                table = render_table(text)
                documents.append(
                    {
                        "doc_id": doc_id,
                        "text": table if table is not None else text,
                        "title": "",
                        "metadata": {
                            "domain": domain,
                            "source": f"ragbench/{source}",
                            "kind": "table" if table is not None else "text",
                        },
                    }
                )
                gold.append(doc_id)
            samples.append(
                {
                    "query_id": f"{prefix}-{q:02d}",
                    "question": row["question"],
                    "reference_answer": row.get("ground_truth") or None,
                    "reference_source": "ragbench_response",
                    "gold_doc_ids": gold,
                    "answerable": True,
                    "metadata": {"domain": domain},
                }
            )
    return documents, samples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples-dir", default="data/ragbench_samples", type=Path)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument("--corpus-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path, help="also write the evaluation set")
    args = parser.parse_args(argv)
    documents, samples = convert(args.samples_dir, args.domains)
    write_jsonl(args.corpus_out, documents)
    print(f"wrote {len(documents)} documents to {args.corpus_out}")
    if args.eval_out:
        write_jsonl(args.eval_out, samples)
        print(f"wrote {len(samples)} questions to {args.eval_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

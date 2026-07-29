"""Batch ingestion script for the kb-rag API.

Walks a directory and uploads every matching file to the ``/api/v1/ingest``
endpoint, printing per-file outcomes and a final summary. The ``--clean`` flag
first deletes existing documents via ``GET/DELETE /api/v1/documents``.

Usage::

    python scripts/seed.py --dir data/raw --api http://localhost:8000

The script uses the ``requests`` library with a 10s timeout per call. It is
imported lazily so that ``--help`` works even when ``requests`` is not yet
installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

INGEST_PATH = "/api/v1/ingest"
DOCUMENTS_PATH = "/api/v1/documents"
TIMEOUT = 10.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed the kb-rag knowledge base by uploading files via the ingest API.",
    )
    parser.add_argument(
        "--dir",
        default="data/raw",
        help="Directory of files to ingest (default: data/raw).",
    )
    parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="Base URL of the kb-rag API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for files to ingest (default: *).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing documents before ingesting.",
    )
    return parser.parse_args(argv)


def clean_existing(api_base: str, session: Any) -> int:
    """Delete existing documents via the API. Returns the count removed.

    Tolerates multiple response shapes from ``GET /api/v1/documents`` (a bare
    list, or ``{"documents": [...]}`` / ``{"items": [...]}`` / ``{"data": [...]}``)
    and silently skips deletion when the API does not expose a ``DELETE``
    endpoint (HTTP 405).
    """
    url = api_base.rstrip("/") + DOCUMENTS_PATH
    deleted = 0
    try:
        resp = session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - network/HTTP errors are non-fatal here
        print(f"[clean] failed to list documents: {exc}", file=sys.stderr)
        return 0

    payload = resp.json()
    if isinstance(payload, dict):
        candidates = (
            payload.get("documents")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []

    for item in candidates:
        doc_id = item.get("doc_id") if isinstance(item, dict) else item
        if not doc_id:
            continue
        try:
            r = session.delete(f"{url}/{doc_id}", timeout=TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            print(f"[clean] error deleting {doc_id}: {exc}", file=sys.stderr)
            continue
        if r.status_code in (200, 204):
            deleted += 1
        elif r.status_code == 405:
            print(
                "[clean] DELETE not supported by API; skipping remaining docs.",
                file=sys.stderr,
            )
            break
        else:
            print(
                f"[clean] could not delete {doc_id}: HTTP {r.status_code}",
                file=sys.stderr,
            )
    return deleted


def ingest_file(api_base: str, file_path: Path, session: Any) -> dict[str, Any]:
    """Upload a single file to the ingest endpoint. Returns parsed JSON."""
    url = api_base.rstrip("/") + INGEST_PATH
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh)}
        resp = session.post(url, files=files, timeout=TIMEOUT)
    if resp.status_code >= 400:
        return {
            "doc_id": None,
            "num_chunks": 0,
            "errors": [f"HTTP {resp.status_code}: {resp.text[:200]}"],
        }
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, optionally clean, then ingest every file."""
    args = parse_args(argv)

    try:
        import requests
    except ImportError:
        print(
            "error: the 'requests' package is required. "
            "Install with: pip install requests",
            file=sys.stderr,
        )
        return 2

    root = Path(args.dir)
    if not root.exists():
        print(f"[seed] directory does not exist: {root}", file=sys.stderr)
        return 2

    files = sorted(p for p in root.glob(args.pattern) if p.is_file())
    if not files:
        print(f"[seed] no files matched {args.pattern!r} in {root}")
        return 0

    session = requests.Session()
    if args.clean:
        n = clean_existing(args.api, session)
        print(f"[seed] cleaned {n} existing document(s)")

    success = 0
    failure = 0
    total_chunks = 0
    for fp in files:
        try:
            result = ingest_file(args.api, fp, session)
        except Exception as exc:  # noqa: BLE001 - never abort the whole batch
            print(f"[seed] {fp.name}: ERROR {exc}")
            failure += 1
            continue

        doc_id = result.get("doc_id")
        num_chunks = int(result.get("num_chunks", 0) or 0)
        errors = result.get("errors") or []
        if errors or doc_id is None:
            failure += 1
            status = "FAIL"
        else:
            success += 1
            total_chunks += num_chunks
            status = "OK"
        print(
            f"[seed] {fp.name}: {status} doc_id={doc_id} "
            f"chunks={num_chunks} errors={len(errors)}"
        )
        for err in errors:
            print(f"        - {err}")

    print(
        f"\n[summary] success={success} failure={failure} "
        f"total_chunks={total_chunks} files_scanned={len(files)}"
    )
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

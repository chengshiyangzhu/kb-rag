"""ChromaDB-backed vector store implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models.document import Chunk, Metadata
from app.observability.logging import get_logger
from app.stores.base import VectorStore

logger = get_logger(__name__)


class ChromaStore(VectorStore):
    """Vector store backed by ``chromadb.PersistentClient``.

    Args:
        path: Filesystem path where Chroma persists its data.
        collection_name: Name of the Chroma collection to use.
        dim: Vector dimensionality (kept for interface symmetry; Chroma infers
            dimensionality from the first upsert).
    """

    def __init__(self, path: str, collection_name: str, dim: int) -> None:
        """Initialize the persistent client and get/create the collection."""
        self._path = path
        self._collection_name = collection_name
        self._dim = int(dim)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @staticmethod
    def _build_where(filters: dict | None) -> dict | None:
        """Translate a filter dict into a Chroma ``where`` clause (best-effort).

        Chroma only supports simple equality on metadata fields. Compound
        filters fall back to the first supported key encountered.
        """
        if not filters:
            return None

        doc_id = filters.get("doc_id")
        if doc_id is not None:
            return {"doc_id": str(doc_id)}

        source = filters.get("source")
        if isinstance(source, str):
            return {"source": source}

        tag = filters.get("tag")
        if isinstance(tag, str):
            return {"tag": tag}

        return None

    @staticmethod
    def _metadata_to_chroma(meta: Metadata) -> dict[str, Any]:
        """Convert a :class:`Metadata` into a JSON-serializable dict for Chroma."""
        out: dict[str, Any] = {
            "chunk_id": meta.doc_id,  # Chroma uses its own id; this is auxiliary
            "source": meta.source,
            "doc_id": meta.doc_id,
            "chunk_index": int(meta.chunk_index),
            "created_at": meta.created_at.isoformat(),
        }
        if meta.page is not None:
            out["page"] = int(meta.page)
        if meta.sheet is not None:
            out["sheet"] = str(meta.sheet)
        if meta.tag:
            out["tag"] = ",".join(meta.tag)
        return out

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert chunks and their vectors into the Chroma collection."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")

        ids = [chunk.id for chunk in chunks]
        embeddings = [list(map(float, v)) for v in vectors]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._metadata_to_chroma(chunk.metadata) for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("upserted chunks", count=len(ids), collection=self._collection_name)

    def search(
        self,
        query_vector: list[float],
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Search the Chroma collection for the most similar chunks."""
        where = self._build_where(filters)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(map(float, query_vector))],
            "n_results": top_n,
        }
        if where is not None:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        results: list[Chunk] = []
        ids_batch = (raw.get("ids") or [[]])[0]
        docs_batch = (raw.get("documents") or [[]])[0]
        metas_batch = (raw.get("metadatas") or [[]])[0]
        dists_batch = (raw.get("distances") or [[]])[0]

        for cid, doc, meta, dist in zip(ids_batch, docs_batch, metas_batch, dists_batch, strict=False):
            meta = meta or {}
            created_at_value = meta.get("created_at")
            try:
                created_at = (
                    datetime.fromisoformat(created_at_value)
                    if isinstance(created_at_value, str)
                    else datetime.utcnow()
                )
            except (TypeError, ValueError):
                created_at = datetime.utcnow()

            tag_value = meta.get("tag")
            tag_list = tag_value.split(",") if isinstance(tag_value, str) and tag_value else []

            metadata = Metadata(
                source=meta.get("source", ""),
                page=meta.get("page"),
                sheet=meta.get("sheet"),
                tag=tag_list,
                created_at=created_at,
                doc_id=meta.get("doc_id", ""),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
            )
            try:
                score = float(dist)
            except (TypeError, ValueError):
                score = 0.0
            results.append(Chunk(id=str(cid), text=doc or "", metadata=metadata, score=score))
        return results

    def delete_by_doc(self, doc_id: str) -> None:
        """Delete all chunks whose ``doc_id`` metadata matches ``doc_id``."""
        self._collection.delete(where={"doc_id": str(doc_id)})
        logger.info("deleted by doc_id", doc_id=doc_id, collection=self._collection_name)

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete the chunks identified by the given chunk ids."""
        if not chunk_ids:
            return
        self._collection.delete(ids=list(chunk_ids))
        logger.info("deleted by chunk_ids", count=len(chunk_ids), collection=self._collection_name)

    def count(self) -> int:
        """Return the number of items in the collection."""
        return int(self._collection.count())

"""Qdrant-backed vector store implementation."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_OID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.models.document import Chunk, Metadata
from app.observability.logging import get_logger
from app.stores.base import VectorStore

logger = get_logger(__name__)


def _to_unix_ts(value: object) -> float:
    """Convert an ISO-8601 string or numeric value to a unix timestamp (float).

    Qdrant's :class:`Range` filter only accepts numeric values, so datetime
    filtering is performed against a numeric ``created_at`` payload field.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"cannot parse ISO timestamp: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise TypeError(f"unsupported time_range value type: {type(value).__name__}")


class QdrantStore(VectorStore):
    """Vector store backed by `Qdrant <https://qdrant.tech>`_.

    Args:
        url: Qdrant REST endpoint, e.g. ``"http://qdrant:6333"``.
        collection_name: Name of the Qdrant collection to use.
        dim: Vector dimensionality. Used only when the collection is first
            created; existing collections keep their original dimensionality.
    """

    def __init__(self, url: str, collection_name: str, dim: int) -> None:
        """Initialize the client and ensure the collection exists."""
        self._url = url
        self._collection_name = collection_name
        self._dim = int(dim)
        self._client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection with HNSW + Cosine config if it is missing."""
        try:
            existing = self._client.get_collection(self._collection_name)
            # Collection exists; trust its configuration.
            logger.debug("qdrant collection exists", collection=self._collection_name, points=existing.points_count)
            return
        except Exception:
            # Collection does not exist (or other transient error); create it.
            logger.info("creating qdrant collection", collection=self._collection_name, dim=self._dim)

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Convert a chunk id into a stable UUID string for Qdrant point storage."""
        return str(uuid5(NAMESPACE_OID, chunk_id))

    @staticmethod
    def _parse_created_at(payload: dict) -> datetime:
        """Reconstruct a timezone-aware datetime from a Qdrant payload.

        Prefers the numeric ``created_at`` unix timestamp (used for Range
        filtering); falls back to the ``created_at_iso`` string and finally
        to ``utcnow`` if neither is present or parseable.
        """
        raw = payload.get("created_at")
        if isinstance(raw, (int, float)):
            try:
                return datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                pass
        iso_value = payload.get("created_at_iso")
        if isinstance(iso_value, str):
            try:
                dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _build_filter(filters: dict | None) -> Filter | None:
        """Translate a filter dict into a Qdrant :class:`Filter` object."""
        if not filters:
            return None

        must: list[FieldCondition] = []

        sources = filters.get("source")
        if sources is not None:
            values = sources if isinstance(sources, list) else [sources]
            must.append(FieldCondition(key="source", match=MatchAny(any=[str(s) for s in values])))

        tags = filters.get("tag")
        if tags is not None:
            values = tags if isinstance(tags, list) else [tags]
            must.append(FieldCondition(key="tag", match=MatchAny(any=[str(t) for t in values])))

        doc_id = filters.get("doc_id")
        if doc_id is not None:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id))))

        time_range = filters.get("time_range")
        if time_range:
            range_kwargs: dict[str, float] = {}
            if time_range.get("gte") is not None:
                range_kwargs["gte"] = _to_unix_ts(time_range["gte"])
            if time_range.get("lte") is not None:
                range_kwargs["lte"] = _to_unix_ts(time_range["lte"])
            if range_kwargs:
                must.append(FieldCondition(key="created_at", range=Range(**range_kwargs)))

        if not must:
            return None
        return Filter(must=must)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert chunks and their vectors into the Qdrant collection."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")

        points: list[PointStruct] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            meta = chunk.metadata
            payload = {
                "chunk_id": chunk.id,
                "source": meta.source,
                "page": meta.page,
                "sheet": meta.sheet,
                "tag": list(meta.tag) if meta.tag else [],
                # Store as numeric unix timestamp so Qdrant Range filters work.
                "created_at": meta.created_at.timestamp(),
                "created_at_iso": meta.created_at.isoformat(),
                "doc_id": meta.doc_id,
                "chunk_index": meta.chunk_index,
                "text": chunk.text,
            }
            if meta.bbox is not None:
                payload["bbox"] = meta.bbox
            points.append(
                PointStruct(id=self._point_id(chunk.id), vector=list(vec), payload=payload)
            )

        self._client.upsert(collection_name=self._collection_name, points=points)
        logger.debug("upserted points", count=len(points), collection=self._collection_name)

    def search(
        self,
        query_vector: list[float],
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Search the collection for the most similar chunks to ``query_vector``."""
        query_filter = self._build_filter(filters)
        hits = self._client.search(
            collection_name=self._collection_name,
            query_vector=list(query_vector),
            limit=top_n,
            query_filter=query_filter,
        )

        results: list[Chunk] = []
        for hit in hits:
            payload = hit.payload or {}
            created_at = self._parse_created_at(payload)

            metadata = Metadata(
                source=payload.get("source", ""),
                page=payload.get("page"),
                sheet=payload.get("sheet"),
                tag=list(payload.get("tag") or []),
                created_at=created_at,
                doc_id=payload.get("doc_id", ""),
                chunk_index=int(payload.get("chunk_index", 0) or 0),
                bbox=payload.get("bbox"),
            )
            results.append(
                Chunk(
                    id=payload.get("chunk_id", str(hit.id)),
                    text=payload.get("text", ""),
                    metadata=metadata,
                    score=float(hit.score) if hit.score is not None else None,
                )
            )
        return results

    def delete_by_doc(self, doc_id: str) -> None:
        """Delete all points whose ``doc_id`` payload matches ``doc_id``."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id)))]
            ),
        )
        logger.info("deleted by doc_id", doc_id=doc_id, collection=self._collection_name)

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete the points identified by the given chunk ids."""
        if not chunk_ids:
            return
        point_ids = [self._point_id(cid) for cid in chunk_ids]
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qmodels.PointIdsList(points=point_ids),
        )
        logger.info("deleted by chunk_ids", count=len(point_ids), collection=self._collection_name)

    def count(self) -> int:
        """Return the number of points in the collection."""
        result = self._client.count(collection_name=self._collection_name, exact=True)
        return int(result.count)

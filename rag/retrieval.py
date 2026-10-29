"""Stufe 3: Grobselektion / Retrieval Stage 1 (Qdrant Vector Search).

Stellt zwei austauschbare Retrieval-Strategien bereit, die für einen Query-Chunk aus
Datei A die K=25 ähnlichsten Kandidaten-Chunks aus Datei B aus Qdrant abrufen.
"""

import re

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models


def _extract_xpath_segments(xpath: str) -> list[str]:
    """Entfernt Positionsindizes wie [1] aus einem XPath und zerlegt ihn in Segmente."""
    cleaned = re.sub(r"\[\d+\]", "", xpath)
    return [seg for seg in cleaned.split("/") if seg]


def _build_hierarchical_filter(
    query_chunk: Document,
    depth_tolerance: int = 1,
) -> models.Filter | None:
    """Erstellt einen Qdrant-Vorfilter auf Basis von parent, depth und XPath-Präfix."""
    meta = query_chunk.metadata
    parent: str = meta.get("parent", "")
    depth: int | None = meta.get("depth")
    xpath: str = meta.get("xpath", "")

    must_conditions: list[models.Condition] = []

    # 1. Baumtiefe einschränken (nutzt INTEGER Payload-Index auf metadata.depth)
    if depth is not None:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.depth",
                range=models.Range(
                    gte=max(0, depth - depth_tolerance),
                    lte=depth + depth_tolerance,
                ),
            )
        )

    # 2. Eltern-Tag einschränken (nutzt KEYWORD Payload-Index auf metadata.parent)
    segments = _extract_xpath_segments(xpath)
    own_tag = segments[-1] if segments else ""
    allowed_parents = [p for p in dict.fromkeys([parent, own_tag]) if p]
    if allowed_parents:
        if len(allowed_parents) == 1 or depth_tolerance == 0:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.parent",
                    match=models.MatchValue(value=allowed_parents[0]),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.parent",
                    match=models.MatchAny(any=allowed_parents),
                )
            )

    # 3. Strukturelles XPath-Präfix einschränken (nutzt TEXT Payload-Index auf metadata.xpath)
    if segments:
        prefix_segments = segments[:-1] if len(segments) > 2 else segments
        xpath_prefix_tokens = " ".join(prefix_segments)
        must_conditions.append(
            models.FieldCondition(
                key="metadata.xpath",
                match=models.MatchText(text=xpath_prefix_tokens),
            )
        )

    if not must_conditions:
        return None
    return models.Filter(must=must_conditions)


def retrieve_unfiltered(
    vectorstore: QdrantVectorStore,
    query_chunk: Document,
    k: int = 25,
) -> list[Document]:
    """Reine unbeschränkte Vektorsuche über alle Chunks von Datei B."""
    return vectorstore.similarity_search(query=query_chunk.page_content, k=k)


def retrieve_hierarchical(
    vectorstore: QdrantVectorStore,
    query_chunk: Document,
    k: int = 25,
) -> list[Document]:
    """Hierarchisch vorgefilterte Vektorsuche über Qdrant Payload-Filter (parent, depth, xpath)."""
    qdrant_filter = _build_hierarchical_filter(query_chunk)
    return vectorstore.similarity_search(
        query=query_chunk.page_content,
        k=k,
        filter=qdrant_filter,
    )


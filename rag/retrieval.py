import re

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models


def retrieve_unfiltered(
    vectorstore: QdrantVectorStore,
    query_chunk: Document,
    k: int = 25,
) -> list[Document]:
    return vectorstore.similarity_search(query=query_chunk.page_content, k=k)


def retrieve_hierarchical(
    vectorstore: QdrantVectorStore,
    query_chunk: Document,
    k: int = 25,
) -> list[Document]:
    meta = query_chunk.metadata
    conditions: list[models.Condition] = []

    if "depth" in meta:
        depth = meta["depth"]
        conditions.append(
            models.FieldCondition(
                key="metadata.depth",
                range=models.Range(gte=max(0, depth - 1), lte=depth + 1),
            )
        )

    if meta.get("parent"):
        conditions.append(
            models.FieldCondition(
                key="metadata.parent",
                match=models.MatchValue(value=meta["parent"]),
            )
        )

    segments = [s for s in re.sub(r"\[\d+\]", "", meta.get("xpath", "")).split("/") if s]
    if segments:
        prefix = " ".join(segments[:-1] if len(segments) > 2 else segments)
        conditions.append(
            models.FieldCondition(
                key="metadata.xpath",
                match=models.MatchText(text=prefix),
            )
        )

    qdrant_filter = models.Filter(must=conditions) if conditions else None
    return vectorstore.similarity_search(
        query=query_chunk.page_content,
        k=k,
        filter=qdrant_filter,
    )

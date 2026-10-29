"""Core-Library `rag-xml-lab`: Zweistufige RAG-Pipeline für den XML-Dokumentenvergleich."""

from rag.chunking import chunk_xml_recursive, chunk_xml_structural
from rag.generation import (
    DIFF_PROMPT,
    DiffResult,
    XmlChange,
    build_diff_chain,
    compare_xml_chunk,
    create_chat_model,
)
from rag.indexing import (
    create_qdrant_client,
    get_embeddings,
    index_xml_chunks,
    setup_payload_indexes,
)
from rag.reranking import rerank_listwise_jina, rerank_pointwise_bge
from rag.retrieval import retrieve_hierarchical, retrieve_unfiltered

__all__ = [
    "DIFF_PROMPT",
    "DiffResult",
    "XmlChange",
    "build_diff_chain",
    "chunk_xml_recursive",
    "chunk_xml_structural",
    "compare_xml_chunk",
    "create_chat_model",
    "create_qdrant_client",
    "get_embeddings",
    "index_xml_chunks",
    "rerank_listwise_jina",
    "rerank_pointwise_bge",
    "retrieve_hierarchical",
    "retrieve_unfiltered",
    "setup_payload_indexes",
]

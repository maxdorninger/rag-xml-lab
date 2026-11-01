import json
import sys
from pathlib import Path
from pprint import pprint

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from rag.chunking import chunk_xml_recursive, chunk_xml_structural
from rag.generation import DiffEntry, compare_xml_chunk
from rag.indexing import index_xml_bge, index_xml_chunks, index_xml_jina, index_xml_qwen
from rag.reranking import (
    rerank_listwise_jina,
    rerank_pointwise_bge,
    rerank_pointwise_qwen,
)
from rag.retrieval import retrieve_hierarchical, retrieve_unfiltered

INDEX_FN = index_xml_jina
# INDEX_FN = index_xml_qwen
# INDEX_FN = index_xml_bge

CHUNK_FN = chunk_xml_structural
# CHUNK_FN = chunk_xml_recursive
CHUNK_MAX_LENGTH = 1500

RETRIEVE_FN = retrieve_hierarchical
# RETRIEVE_FN = retrieve_unfiltered

RERANK_FN = rerank_listwise_jina
# RERANK_FN = rerank_pointwise_bge
# RERANK_FN = rerank_pointwise_qwen


RETRIEVE_TOP_K = 25  # how many chunks to retrieve from qdrant
RERANK_TOP_N = 3  # how many chunks to embed into the context after reranking

QDRANT_LOCATION = ":memory:"  # "http://localhost:6333"
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434"

BASE_DIR = Path(__file__).parent
FILE_A = str(BASE_DIR / "sample_data" / "file_a.xml")
FILE_B = str(BASE_DIR / "sample_data" / "file_b.xml")


def run_pipeline(file_a: str = FILE_A, file_b: str = FILE_B) -> list[dict]:
    chunks_a = CHUNK_FN(file_a)
    chunks_b = CHUNK_FN(file_b)
    print(f"[Stage 1] File A = {len(chunks_a)} chunks, File B = {len(chunks_b)} chunks")

    print("[Stage 2] Saving Chunks in Qdrant...")
    vectorstore = INDEX_FN(
        chunks=chunks_b,
        collection_name="xml_file_b",
        location=QDRANT_LOCATION,
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0)

    # LCEL pipeline: query chunk -> retrieve candidates -> rerank -> LLM comparison
    compare_chain = (
        {"chunk": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            candidates=lambda x: RETRIEVE_FN(vectorstore, x["chunk"], k=RETRIEVE_TOP_K)
        )
        | RunnablePassthrough.assign(
            top_3=lambda x: RERANK_FN(x["chunk"], x["candidates"], top_n=RERANK_TOP_N)
        )
        | RunnableLambda(lambda x: compare_xml_chunk(x["chunk"], x["top_3"], llm=llm))
    )

    diffs: list[DiffEntry] = []

    for i, query_chunk in enumerate(chunks_a, start=1):
        print(f"[Stage 3] Chunk {i}/{len(chunks_a)}: {query_chunk.metadata['xpath']}")
        result = compare_chain.invoke(query_chunk)
        pprint(result)
        diffs.extend(result)
    return [d.model_dump() for d in diffs]


if __name__ == "__main__":
    pprint(json.dumps(run_pipeline(), indent=2, ensure_ascii=False))

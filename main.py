import json
import sys
from pathlib import Path

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
#INDEX_FN = index_xml_qwen
#INDEX_FN = index_xml_bge

CHUNK_FN = chunk_xml_structural
#CHUNK_FN = chunk_xml_recursive   

RETRIEVE_FN = retrieve_hierarchical
#RETRIEVE_FN = retrieve_unfiltered

RERANK_FN = rerank_listwise_jina
#RERANK_FN = rerank_pointwise_bge
#RERANK_FN = rerank_pointwise_qwen


RETRIEVE_TOP_K = 25 # how many chunks to retrieve from qdrant
RERANK_TOP_N = 3 # how many chunks to embed into the context after reranking

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
    diffs: list[DiffEntry] = []
    matched_xpaths_b: set[str] = set()

    for i, query_chunk in enumerate(chunks_a, start=1):
        print(f"[Stage 3] Chunk {i}/{len(chunks_a)}: {query_chunk.metadata['xpath']}")
        candidates = RETRIEVE_FN(vectorstore, query_chunk, k=RETRIEVE_TOP_K)
        top_3 = RERANK_FN(query_chunk, candidates, top_n=RERANK_TOP_N)
        result = compare_xml_chunk(query_chunk, top_3, llm=llm)

        if result.matched_xpath_b:
            matched_xpaths_b.add(result.matched_xpath_b)
        diffs.extend(result.diffs)

    # Emit 'Added' for chunks in File B that had no counterpart in File A
    for chunk_b in chunks_b:
        if chunk_b.metadata["xpath"] not in matched_xpaths_b:
            start = chunk_b.metadata["start_line"]
            for offset, line in enumerate(chunk_b.page_content.splitlines()):
                if line.strip():
                    diffs.append(
                        DiffEntry(
                            position=start + offset,
                            status="Added",
                            content=line.strip(),
                        )
                    )

    return [d.model_dump() for d in diffs]


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2, ensure_ascii=False))

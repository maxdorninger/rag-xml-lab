"""Showcase-Skript für `rag-xml-lab`: Zweistufige RAG-Pipeline zum XML-Dokumentenvergleich.

Demonstriert das Baukasten-Prinzip: Jede Pipeline-Strategie (Chunking, Retrieval,
Reranking, Qdrant-Backend, LLM) lässt sich in einer einzigen Konfigurationszeile austauschen.
"""

from pathlib import Path

from rag.chunking import chunk_xml_recursive, chunk_xml_structural
from rag.generation import compare_xml_chunk, create_chat_model
from rag.indexing import index_xml_chunks
from rag.reranking import rerank_listwise_jina, rerank_pointwise_bge
from rag.retrieval import retrieve_hierarchical, retrieve_unfiltered

CHUNK_FN = chunk_xml_structural          # Alternativ: chunk_xml_recursive
RETRIEVE_FN = retrieve_hierarchical      # Alternativ: retrieve_unfiltered
RERANK_FN = rerank_listwise_jina         # Alternativ: rerank_pointwise_bge
RETRIEVE_TOP_K = 25
RERANK_TOP_N = 3

QDRANT_LOCATION = ":memory:"             # Alternativ: "./qdrant_data" oder "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434"

BASE_DIR = Path(__file__).parent
FILE_A = str(BASE_DIR / "sample_data" / "file_a.xml")
FILE_B = str(BASE_DIR / "sample_data" / "file_b.xml")


def run_pipeline() -> None:
    print("=" * 80)
    print("rag pipeline config:")
    print(f"  Chunking  : {CHUNK_FN.__name__}")
    print(f"  Retrieval : {RETRIEVE_FN.__name__} (K=25)")
    print(f"  Reranking : {RERANK_FN.__name__} (Top-3)")
    print(f"  Qdrant    : {QDRANT_LOCATION}")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Stufe 1: XML-Preprocessing & Chunking (lxml.etree -> list[Document])
    # ------------------------------------------------------------------------
    chunks_a = CHUNK_FN(FILE_A)
    chunks_b = CHUNK_FN(FILE_B)
    print(f"\n[Stufe 1] Chunking abgeschlossen: Datei A = {len(chunks_a)} Chunks, Datei B = {len(chunks_b)} Chunks")

    # ------------------------------------------------------------------------
    # Stufe 2: Enrichment & Vector Indexing von Datei B in Qdrant
    # ------------------------------------------------------------------------
    print(f"[Stufe 2] Indiziere Datei B mit '{EMBEDDING_MODEL}' in Qdrant ({QDRANT_LOCATION}) ...")
    vectorstore = index_xml_chunks(
        chunks=chunks_b,
        collection_name="xml_file_b",
        location=QDRANT_LOCATION,
        embedding_model=EMBEDDING_MODEL,
    )

    llm = create_chat_model(model=OLLAMA_MODEL, base_url=OLLAMA_URL)


    for chunk_index, query_chunk in enumerate(chunks_a, start=1):
        xpath_a = query_chunk.metadata["xpath"]
        depth_a = query_chunk.metadata["depth"]
        parent_a = query_chunk.metadata["parent"]

        print("\n" + "-" * 78)
        print(f"Prüfe Chunk {chunk_index}/{len(chunks_a)} aus Datei A: {xpath_a} (depth={depth_a}, parent='{parent_a}')")

        # --------------------------------------------------------------------
        # Stufe 3: Retrieval Stage 1 (K=25)
        # --------------------------------------------------------------------
        candidates = RETRIEVE_FN(vectorstore, query_chunk, k=RETRIEVE_TOP_K)
        print(f"  [Stufe 3] Grobselektion ({RETRIEVE_FN.__name__}): {len(candidates)} Kandidaten geladen")

        # --------------------------------------------------------------------
        # Stufe 4: Reranking Stage 2 (Top 3)
        # --------------------------------------------------------------------
        top_3 = RERANK_FN(query_chunk, candidates, top_n=RERANK_TOP_N)
        print(f"  [Stufe 4] Top-{len(top_3)} nach Reranking ({RERANK_FN.__name__}):")
        for rank, cand in enumerate(top_3, start=1):
            score = cand.metadata.get("rerank_score", 0.0)
            print(f"            #{rank} score={score:.4f} | xpath={cand.metadata.get('xpath')}")

        # --------------------------------------------------------------------
        # Stufe 5: Augmentation & Generation (Structured Output DiffResult)
        # --------------------------------------------------------------------

        diff = compare_xml_chunk(query_chunk, top_3, llm=llm)
        print(
            f"  [Stufe 5] LLM-Diff: status={diff.status} | has_changes={diff.has_changes} "
            f"| matched_xpath_b={diff.matched_xpath_b}"
        )
        print(f"            Summary: {diff.summary}")
        for change in diff.changes:
            print(
                f"            - [{change.change_type}] {change.target_path}: "
                f"{change.old_value!r} -> {change.new_value!r} ({change.explanation})"
            )
        

if __name__ == "__main__":
    run_pipeline()


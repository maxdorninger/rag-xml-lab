from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from transformers import AutoModel

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
JINA_RERANKER_MODEL = "jinaai/jina-reranker-v3.5"


def _with_rerank_score(doc: Document, score: float) -> Document:
    return Document(
        page_content=doc.page_content,
        metadata={**doc.metadata, "rerank_score": float(score)},
    )


def rerank_pointwise_bge(
    query_chunk: Document,
    candidates: list[Document],
    top_n: int = 3,
    model_name: str = BGE_RERANKER_MODEL,
) -> list[Document]:
    if not candidates:
        return []

    reranker = CrossEncoder(model_name)
    pairs = [(query_chunk.page_content, cand.page_content) for cand in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored_docs = [
        _with_rerank_score(cand, float(score))
        for cand, score in zip(candidates, scores, strict=True)
    ]
    scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return scored_docs[:top_n]


def rerank_listwise_jina(
    query_chunk: Document,
    candidates: list[Document],
    top_n: int = 3,
    model_name: str = JINA_RERANKER_MODEL,
) -> list[Document]:
    if not candidates:
        return []

    model = AutoModel.from_pretrained(
        model_name,
        dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    doc_texts = [cand.page_content for cand in candidates]

    ranked_items = model.rerank(
        query=query_chunk.page_content,
        documents=doc_texts,
        top_n=min(top_n, len(candidates)),
    )
    return [
        _with_rerank_score(candidates[int(item["index"])], float(item["relevance_score"]))
        for item in ranked_items
    ]
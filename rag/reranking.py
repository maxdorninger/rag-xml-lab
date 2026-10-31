import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
JINA_RERANKER_MODEL = "jinaai/jina-reranker-v3.5"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"


def rerank_pointwise_bge(
    query_chunk: Document,
    candidates: list[Document],
    top_n: int = 3,
) -> list[Document]:
    if not candidates:
        return []

    reranker = CrossEncoder(BGE_RERANKER_MODEL)
    pairs = [(query_chunk.page_content, cand.page_content) for cand in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored_docs = [
        Document(page_content=cand.page_content, metadata={**cand.metadata, "rerank_score": float(score)})
        for cand, score in zip(candidates, scores, strict=True)
    ]
    scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return scored_docs[:top_n]


def rerank_pointwise_qwen(
    query_chunk: Document,
    candidates: list[Document],
    top_n: int = 3,
) -> list[Document]:
    if not candidates:
        return []

    # des kann man weg machen wenn es deployed wird, weil mps is macos
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(QWEN_RERANKER_MODEL)
    if tokenizer is None:
        raise ValueError("Failed to load tokenizer.")

    model = AutoModelForCausalLM.from_pretrained(QWEN_RERANKER_MODEL, dtype="auto")
    model.to(device=device)  # ty: ignore[missing-argument]
    model.eval()

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instruction = "Given an XML chunk from File A, retrieve the matching XML chunk from File B."

    no_id = tokenizer.convert_tokens_to_ids("no")
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    scored_docs: list[Document] = []

    with torch.no_grad():
        for cand in candidates:
            prompt = (
                f"{prefix}<Instruct>: {instruction}\n"
                f"<Query>: {query_chunk.page_content}\n"
                f"<Document>: {cand.page_content}{suffix}"
            )
            inputs = tokenizer(prompt, truncation=True, max_length=8192, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0, -1, [no_id, yes_id]]
            score = torch.nn.functional.log_softmax(logits, dim=0)[1].exp().item()
            scored_docs.append(
                Document(page_content=cand.page_content, metadata={**cand.metadata, "rerank_score": float(score)})
            )

    scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return scored_docs[:top_n]


def rerank_listwise_jina(
    query_chunk: Document,
    candidates: list[Document],
    top_n: int = 3,
) -> list[Document]:
    if not candidates:
        return []

    model = AutoModel.from_pretrained(JINA_RERANKER_MODEL, dtype="auto", trust_remote_code=True)
    model.eval()

    ranked_items = model.rerank(
        query=query_chunk.page_content,
        documents=[cand.page_content for cand in candidates],
        top_n=min(top_n, len(candidates)),
    )
    return [
        Document(
            page_content=candidates[int(item["index"])].page_content,
            metadata={**candidates[int(item["index"])].metadata, "rerank_score": float(item["relevance_score"])},
        )
        for item in ranked_items
    ]

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
JINA_EMBEDDING_MODEL = "jinaai/jina-code-embeddings-1.5b"
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def index_xml_chunks(
    chunks: list[Document],
    collection_name: str,
    location: str,
    embedding_model: str,
    use_sparse_bm25: bool,
) -> QdrantVectorStore:
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    emb = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"trust_remote_code": True, "device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25") if use_sparse_bm25 else None
    client = QdrantClient(location=location)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    vector_dim = len(emb.embed_query("probe"))
    sparse_config = (
        {"langchain-sparse": models.SparseVectorParams()} if use_sparse_bm25 else None
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dim, distance=models.Distance.COSINE
        ),
        sparse_vectors_config=sparse_config,
    )

    for field, schema in [
        ("metadata.parent", models.PayloadSchemaType.KEYWORD),
        ("metadata.depth", models.PayloadSchemaType.INTEGER),
        ("metadata.xpath", models.PayloadSchemaType.TEXT),
        ("metadata.doc_id", models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=collection_name, field_name=field, field_schema=schema
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=emb,
        sparse_embedding=sparse_emb,
        retrieval_mode=RetrievalMode.HYBRID if use_sparse_bm25 else RetrievalMode.DENSE,
    )
    if chunks:
        vectorstore.add_documents(chunks)

    return vectorstore


def index_xml_bge(
    chunks: list[Document],
    collection_name: str = "xml_comparison_b",
    location: str = ":memory:",
) -> QdrantVectorStore:
    return index_xml_chunks(
        chunks,
        collection_name=collection_name,
        location=location,
        embedding_model=BGE_EMBEDDING_MODEL,
        use_sparse_bm25=False,
    )


def index_xml_jina(
    chunks: list[Document],
    collection_name: str = "xml_comparison_b",
    location: str = ":memory:",
) -> QdrantVectorStore:
    return index_xml_chunks(
        chunks,
        collection_name=collection_name,
        location=location,
        embedding_model=JINA_EMBEDDING_MODEL,
        use_sparse_bm25=True,
    )


def index_xml_qwen(
    chunks: list[Document],
    collection_name: str = "xml_comparison_b",
    location: str = ":memory:",
) -> QdrantVectorStore:
    return index_xml_chunks(
        chunks,
        collection_name=collection_name,
        location=location,
        embedding_model=QWEN_EMBEDDING_MODEL,
        use_sparse_bm25=False,
    )

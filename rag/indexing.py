from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def setup_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.parent",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.depth",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.xpath",
        field_schema=models.PayloadSchemaType.TEXT,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def index_xml_chunks(
    chunks: list[Document],
    collection_name: str = "xml_comparison_b",
    location: str = ":memory:",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> QdrantVectorStore:
    """stores all XML chunks with metadata (xpath, tree depth, parent element, and doc_id) in Qdrant"""
    emb = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    client = QdrantClient(location=location)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    vector_dim = len(emb.embed_query("xml_dimension_probe"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=models.Distance.COSINE,
        ),
    )

    setup_payload_indexes(client, collection_name)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=emb,
    )
    if chunks:
        vectorstore.add_documents(chunks)

    return vectorstore

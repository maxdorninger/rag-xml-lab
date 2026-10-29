"""Stufe 5: Augmentation & Generation (LCEL + Structured Output).

Vergleicht einen Quell-Chunk aus XML-Datei A mit den Top-3-Kandidaten aus XML-Datei B
über ein festes ChatPromptTemplate und gibt ein deterministisches `DiffResult` zurück.
"""

from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class XmlChange(BaseModel):
    """Einzelne festgestellte Änderung auf Attribut-, Text- oder Strukturebene."""

    change_type: Literal["added", "removed", "modified", "moved"] = Field(
        description="Art der Änderung zwischen Datei A und Datei B."
    )
    target_path: str = Field(
        description="Betroffener relativer Pfad, Tag oder Attributname innerhalb des Chunks."
    )
    old_value: str | None = Field(
        default=None,
        description="Bisheriger Wert in Datei A (None bei neu hinzugefügten Elementen).",
    )
    new_value: str | None = Field(
        default=None,
        description="Neuer Wert in Datei B (None bei entfernten Elementen).",
    )
    explanation: str = Field(
        description="Kurze, präzise Beschreibung der Änderung."
    )


class DiffResult(BaseModel):
    """Deterministisches Ergebnisschema des XML-Chunk-Vergleichs."""

    has_changes: bool = Field(
        description="True, wenn semantische, strukturelle oder wertbezogene Unterschiede vorliegen."
    )
    status: Literal["identical", "modified", "deleted_in_b"] = Field(
        description="Gesamtstatus des Quell-Chunks aus Datei A im Abgleich mit Datei B."
    )
    source_xpath: str = Field(
        description="XPath des geprüften Quell-Chunks in Datei A."
    )
    matched_xpath_b: str | None = Field(
        default=None,
        description="XPath des korrespondierenden Kandidaten in Datei B (oder None, falls gelöscht).",
    )
    summary: str = Field(
        description="Prägnante Gesamtzusammenfassung des Vergleichs."
    )
    changes: list[XmlChange] = Field(
        default_factory=list,
        description="Liste aller konkret erkannten Einzeländerungen.",
    )


DIFF_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Du bist ein präzises Analyse-System für den strukturellen und inhaltlichen "
                "Vergleich von XML-Dokumenten.\n"
                "Du erhältst einen Quell-XML-Chunk aus Datei A (inklusive XPath) sowie die "
                "Top-3 ähnlichsten Kandidaten-Chunks aus Datei B.\n"
                "Aufgaben:\n"
                "1. Identifiziere, welcher Kandidat aus Datei B fachlich/strukturell dem "
                "Quell-Chunk aus Datei A entspricht (anhand von IDs, Tags, XPath und Inhalt).\n"
                "2. Entspricht keiner der Kandidaten demselben fachlichen Objekt, setze "
                "`status='deleted_in_b'`, `has_changes=True` und `matched_xpath_b=None`.\n"
                "3. Ist der passende Kandidat inhaltlich und strukturell identisch (reine "
                "Whitespace-Unterschiede ignorieren), setze `status='identical'` und `has_changes=False`.\n"
                "4. Liegen Abweichungen bei Attributen, Textwerten oder Kindelementen vor, "
                "setze `status='modified'`, `has_changes=True` und liste jede Änderung exakt auf."
            ),
        ),
        (
            "human",
            (
                "### Quell-Chunk aus Datei A\n"
                "- XPath: {source_xpath}\n"
                "- Tiefe: {source_depth} | Parent: {source_parent}\n"
                "```xml\n{source_xml}\n```\n\n"
                "### Top-3 Kandidaten aus Datei B\n"
                "{candidates_formatted}\n\n"
                "Führe nun den strukturierten XML-Diff durch."
            ),
        ),
    ]
)


def format_candidates(candidates: list[Document]) -> str:
    """Formatiert die Top-3-Kandidaten samt Metadaten und Rerank-Score für den Prompt."""
    if not candidates:
        return "(Keine Kandidaten in Datei B gefunden.)"

    blocks: list[str] = []
    for rank, doc in enumerate(candidates, start=1):
        xpath = doc.metadata.get("xpath", "unbekannt")
        depth = doc.metadata.get("depth", "?")
        parent = doc.metadata.get("parent", "")
        score = doc.metadata.get("rerank_score")
        score_str = f"{score:.4f}" if isinstance(score, float | int) else "n/a"
        blocks.append(
            f"#### Kandidat #{rank}\n"
            f"- XPath: {xpath} | Tiefe: {depth} | Parent: {parent} | Rerank-Score: {score_str}\n"
            f"```xml\n{doc.page_content}\n```"
        )
    return "\n\n".join(blocks)


def create_chat_model(
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    provider: Literal["ollama", "openai"] = "ollama",
    temperature: float = 0.0,
) -> BaseChatModel:
    """Erstellt das Chat-Modell (Standard: ChatOllama; alternativ OpenAI-Endpunkt z. B. llama-server)."""
    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=base_url,
            format="json", 
            temperature=temperature,
        )
    # Kompatibel mit lokalem `llama-server` / vLLM über OpenAI-kompatiblen Endpunkt
    return init_chat_model(
        model=model,
        model_provider="openai",
        base_url=base_url,
        api_key="not-needed",
        temperature=temperature,
    )


def build_diff_chain(llm: BaseChatModel | None = None) -> Runnable[dict[str, object], DiffResult]:
    """Baut die LCEL-Pipeline `DIFF_PROMPT | llm.with_structured_output(DiffResult)`."""
    chat_model = llm or create_chat_model()
    structured_llm = chat_model.with_structured_output(DiffResult)
    return cast(Runnable[dict[str, object], DiffResult], DIFF_PROMPT | structured_llm)


def compare_xml_chunk(
    query_chunk: Document,
    top_candidates: list[Document],
    llm: BaseChatModel | None = None,
) -> DiffResult:
    """Führt Stufe 5 für einen Quell-Chunk aus Datei A und seine Top-3-Kandidaten aus Datei B aus."""
    chain = build_diff_chain(llm)
    return chain.invoke(
        {
            "source_xpath": query_chunk.metadata.get("xpath", ""),
            "source_depth": query_chunk.metadata.get("depth", 0),
            "source_parent": query_chunk.metadata.get("parent", ""),
            "source_xml": query_chunk.page_content,
            "candidates_formatted": format_candidates(top_candidates),
        }
    )


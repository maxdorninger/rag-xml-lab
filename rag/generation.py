from typing import Literal, cast

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class DiffEntry(BaseModel):
    position: int = Field(
        description="1-based line number (File A for Removed, File B for Added/Modified)."
    )
    status: Literal["Added", "Removed", "Modified"]
    content: str = Field(description="Trimmed XML line without '[Line N]' prefix.")


class DiffEntries(BaseModel):
    entries: list[DiffEntry] = Field(description="A collection of difference entries")


DIFF_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an XML diff engine comparing one source chunk from File A against up to 3 "
                "candidate chunks from File B. Each line starts with `[Line N]` ONLY as a line-number "
                "reference — ignore line-number shifts when comparing XML content.\n\n"
                "Rules:\n"
                "1. Find the single candidate in File B representing the same entity (match `id`, `sku`, "
                "`name`, or tag structure; never match elements with different IDs) and set `matched_xpath_b`.\n"
                "2. If no candidate matches, set `matched_xpath_b=null` and return `Removed` for every "
                "line of the File A chunk using File A's line numbers.\n"
                "3. Compare ONLY the File A chunk with its single matched File B candidate:\n"
                "   - Identical XML content -> `diffs=[]`.\n"
                "   - Changed XML line -> `status='Modified'`, `position` = File B line, `content` = File B XML.\n"
                "   - Inserted XML line in File B -> `status='Added'`, `position` = File B line, `content` = File B XML.\n"
                "   - Deleted XML line from File A -> `status='Removed'`, `position` = File A line, `content` = File A XML.\n"
                "4. `content` must be the trimmed XML line WITHOUT `[Line N]`."
            ),
        ),
        (
            "human",
            (
                "### Source Chunk (File A) | XPath: {source_xpath}\n"
                "```xml\n{source_xml}\n```\n\n"
                "### Top-3 Candidates (File B)\n"
                "{candidates_xml}"
            ),
        ),
    ]
)


def _numbered_xml(doc: Document) -> str:
    start = doc.metadata.get("start_line", 1)
    return "\n".join(
        f"[Line {start + i}] {line.strip()}"
        for i, line in enumerate(doc.page_content.splitlines())
        if line.strip()
    )


def compare_xml_chunk(
    query_chunk: Document,
    top_candidates: list[Document],
    llm: ChatOllama,
) -> list[DiffEntry]:
    candidates_xml = "\n\n".join(
        f"Candidate #{i} (XPath: {c.metadata.get('xpath')}):\n```xml\n{_numbered_xml(c)}\n```"
        for i, c in enumerate(top_candidates, start=1)
    )
    chain = DIFF_PROMPT | llm.with_structured_output(DiffEntries)
    result: DiffEntries = chain.invoke(
        {
            "source_xpath": query_chunk.metadata.get("xpath", ""),
            "source_xml": _numbered_xml(query_chunk),
            "candidates_xml": candidates_xml or "(none)",
        }
    )  # ty: ignore[invalid-assignment]
    return result.entries

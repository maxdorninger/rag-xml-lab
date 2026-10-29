from copy import deepcopy
from pathlib import Path

from langchain_core.documents import Document
from lxml import etree


def _local_tag(element: etree._Element) -> str:
    return etree.QName(element).localname


def _get_depth(element: etree._Element) -> int:
    return sum(1 for _ in element.iterancestors())


def _get_parent_tag(element: etree._Element) -> str:
    parent = element.getparent()
    return _local_tag(parent) if parent is not None else ""


def _element_children(element: etree._Element) -> list[etree._Element]:
    return [child for child in element if isinstance(child.tag, str)]


def _serialize_xml(element: etree._Element) -> str:
    return etree.tostring(
        element,
        encoding="unicode",
        pretty_print=True,
        with_tail=False,
    ).strip()


def _make_shell(element: etree._Element) -> etree._Element:
    shell = etree.Element(element.tag, attrib=dict(element.attrib), nsmap=element.nsmap)
    if element.text and element.text.strip():
        shell.text = element.text
    return shell


def _build_document(
    content_element: etree._Element,
    source_node: etree._Element,
    tree: etree._ElementTree,
    doc_id: str,
) -> Document:
    """saves an XML tag as a langchain document"""
    return Document(
        page_content=_serialize_xml(content_element),
        metadata={
            "xpath": tree.getpath(source_node),
            "depth": _get_depth(source_node),
            "parent": _get_parent_tag(source_node),
            "doc_id": doc_id,
        },
    )


def chunk_xml_structural(file_path: str, target_depth: int = 2) -> list[Document]:
    """Zerlegt eine XML-Datei anhand fester struktureller Subtrees (Standard: Tiefe 2).

    Endet ein Teilbaum bereits oberhalb von `target_depth` als Blattknoten, wird er
    ebenfalls als Chunk übernommen, sodass keine XML-Inhalte verloren gehen.
    """
    parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
    tree = etree.parse(file_path, parser)
    root = tree.getroot()
    doc_id = Path(file_path).name
    chunks: list[Document] = []

    def _walk(node: etree._Element, depth: int) -> None:
        children = _element_children(node)
        if depth >= target_depth or not children:
            chunks.append(_build_document(node, node, tree, doc_id))
            return
        for child in children:
            _walk(child, depth + 1)

    _walk(root, depth=0)
    return chunks


def chunk_xml_recursive(file_path: str, max_chars: int = 1500) -> list[Document]:
    """Adaptives Recursive Chunking mit Erhalt des umschließenden XML-Eltern-Wrappers.

    Prüft rekursiv die Zeichenlänge jedes XML-Knotens. Überschreitet ein Knoten
    `max_chars`, wird er entlang seiner Kindelemente aufgeteilt, wobei das
    umschließende Elternelement (inkl. Attributen) als valider XML-Wrapper erhalten bleibt.
    Überschreitet bereits ein einzelnes Kindelement im Wrapper das Limit, steigt der
    Algorithmus rekursiv in dieses Kindelement ab.
    """
    parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
    tree = etree.parse(file_path, parser)
    root = tree.getroot()
    doc_id = Path(file_path).name

    def _split_node(node: etree._Element) -> list[Document]:
        serialized = _serialize_xml(node)
        if len(serialized) <= max_chars:
            return [_build_document(node, node, tree, doc_id)]

        children = _element_children(node)
        if not children:
            # Atomarer Blattknoten ohne weitere Kindelemente bleibt valides XML
            return [_build_document(node, node, tree, doc_id)]

        result: list[Document] = []
        current_wrapper = _make_shell(node)
        has_items_in_wrapper = False

        for child in children:
            single_test = _make_shell(node)
            single_test.append(deepcopy(child))

            # Ist ein einzelnes Kind selbst zu groß -> aktuellen Wrapper leeren & rekursiv absteigen
            if len(_serialize_xml(single_test)) > max_chars:
                if has_items_in_wrapper:
                    result.append(_build_document(current_wrapper, node, tree, doc_id))
                    current_wrapper = _make_shell(node)
                    has_items_in_wrapper = False
                result.extend(_split_node(child))
                continue

            # Prüfen, ob das Kind noch in den laufenden Eltern-Wrapper passt
            candidate_wrapper = deepcopy(current_wrapper)
            candidate_wrapper.append(deepcopy(child))

            if has_items_in_wrapper and len(_serialize_xml(candidate_wrapper)) > max_chars:
                result.append(_build_document(current_wrapper, node, tree, doc_id))
                current_wrapper = _make_shell(node)
                current_wrapper.append(deepcopy(child))
                has_items_in_wrapper = True
            else:
                current_wrapper = candidate_wrapper
                has_items_in_wrapper = True

        if has_items_in_wrapper:
            result.append(_build_document(current_wrapper, node, tree, doc_id))

        return result

    return _split_node(root)


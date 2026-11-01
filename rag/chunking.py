from copy import deepcopy
from pathlib import Path

from langchain_core.documents import Document
from lxml import etree


def _to_document(
    content_el: etree._Element,
    source_el: etree._Element,
    tree: etree._ElementTree,
    doc_id: str,
) -> Document:
    parent = source_el.getparent()
    return Document(
        page_content=etree.tostring(content_el, encoding="unicode", with_tail=False).strip(),
        metadata={
            "xpath": tree.getpath(source_el),
            "depth": sum(1 for _ in source_el.iterancestors()),
            "parent": etree.QName(parent).localname if parent is not None else "",
            "doc_id": doc_id,
            "start_line": source_el.sourceline or 1,
        },
    )


def chunk_xml_structural(file_path: str, target_depth: int = 2) -> list[Document]:
    tree = etree.parse(file_path)
    doc_id = Path(file_path).name
    chunks: list[Document] = []

    def walk(node: etree._Element, depth: int) -> None:
        children = [c for c in node if isinstance(c.tag, str)]
        if depth >= target_depth or not children:
            chunks.append(_to_document(node, node, tree, doc_id))
            return
        for child in children:
            walk(child, depth + 1)

    walk(tree.getroot(), depth=0)
    return chunks


def chunk_xml_recursive(file_path: str, max_chars: int) -> list[Document]:
    tree = etree.parse(file_path)
    doc_id = Path(file_path).name

    def split_node(node: etree._Element) -> list[Document]:
        xml_str = etree.tostring(node, encoding="unicode", with_tail=False).strip()
        children = [c for c in node if isinstance(c.tag, str)]
        if len(xml_str) <= max_chars or not children:
            return [_to_document(node, node, tree, doc_id)]

        result: list[Document] = []
        wrapper = etree.Element(node.tag, attrib=dict(node.attrib), nsmap=node.nsmap)

        for child in children:
            single = etree.Element(node.tag, attrib=dict(node.attrib), nsmap=node.nsmap)
            single.append(deepcopy(child))

            if len(etree.tostring(single, encoding="unicode", with_tail=False)) > max_chars:
                if len(wrapper):
                    result.append(_to_document(wrapper, node, tree, doc_id))
                    wrapper = etree.Element(node.tag, attrib=dict(node.attrib), nsmap=node.nsmap)
                result.extend(split_node(child))
                continue

            candidate = deepcopy(wrapper)
            candidate.append(deepcopy(child))

            if len(wrapper) and len(etree.tostring(candidate, encoding="unicode", with_tail=False)) > max_chars:
                result.append(_to_document(wrapper, node, tree, doc_id))
                wrapper = single
            else:
                wrapper = candidate

        if len(wrapper):
            result.append(_to_document(wrapper, node, tree, doc_id))

        return result

    return split_node(tree.getroot())

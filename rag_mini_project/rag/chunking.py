"""parent를 child 청크로 나눈다(split_parents). parent가 없는 문서는 쪽 단위로 바로 나눈다(split_pages)."""
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .md_parser import PageText, Parent

# 문단(항·호) → 줄 → 문장 → 단어 순으로 자를 곳을 찾는다
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50

    @property
    def name(self) -> str:
        return f"c{self.chunk_size}_o{self.chunk_overlap}"


def split_parents(parents: list[Parent], cfg: ChunkConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",   # ". "는 앞 청크 끝에 남긴다
        add_start_index=True,
    )
    docs: list[Document] = []
    for p in parents:
        pieces = splitter.create_documents([p.content])
        for i, d in enumerate(pieces):
            start = d.metadata["start_index"]
            end = start + len(d.page_content) - 1
            meta = {
                "chunk_id": f"{p.parent_id}::{cfg.name}::{i:02d}",
                "parent_id": p.parent_id,
                "doc_id": p.doc_id,
                "doc_type": p.doc_type,
                "source_file": p.source_file,
                "doc_title": p.doc_title,
                "heading": p.heading,
                "article_no": p.article_no,
                "article_title": p.article_title,
                "chunk_index": i,
                "chunk_count": len(pieces),
                "start_index": start,
                "page_start": p.page_at(start),
                "page_end": p.page_at(end),
                "source_url": p.source_url,
                "source_id": p.source_id,          # 골든셋 expected_source_ids와 비교
                "source_ids": p.source_ids or ([p.source_id] if p.source_id else None),
                "source_doc_id": p.source_doc_id,
                "chunk_config": cfg.name,
            }
            # Chroma 메타데이터에는 None을 넣을 수 없다
            docs.append(Document(page_content=d.page_content,
                                 metadata={k: v for k, v in meta.items() if v is not None}))
    return docs


def split_pages(pages: list[PageText], cfg: ChunkConfig) -> list[Document]:
    """parent 없이 쪽마다 나눈다. 청크가 쪽을 넘지 않아 근거 ID(…-pdf-N)가 청크마다 하나로 정해진다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",
        add_start_index=True,
    )
    docs: list[Document] = []
    for p in pages:
        pieces = splitter.create_documents([p.content])
        for i, d in enumerate(pieces):
            docs.append(Document(page_content=d.page_content, metadata={
                "chunk_id": f"{p.doc_id}::p{p.page:03d}::{cfg.name}::{i:02d}",
                "doc_id": p.doc_id,
                "doc_type": p.doc_type,
                "source_file": p.source_file,
                "chunk_index": i,
                "chunk_count": len(pieces),
                "start_index": d.metadata["start_index"],
                "page_start": p.page,
                "page_end": p.page,
                "source_id": p.source_id,
                "source_doc_id": p.source_doc_id,
                "chunk_config": cfg.name,
            }))
    return docs

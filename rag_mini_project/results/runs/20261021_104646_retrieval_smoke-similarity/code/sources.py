"""근거 ID(source_id) → 원문. 청킹 방식과 상관없이 정답 근거 단위의 원문을 돌려준다(평가용).

- 약관: 조항(kofia-terms-…-NNN), 용어집: 용어(fsc-glossary-dicId) → md_parser의 parent 원문
- 가이드북·추가 문서: PDF 쪽(…-pdf-N) → 쪽 원문
"""
from functools import cache

from .config import MD_DIR
from .md_parser import load_pages, load_parents

PARENT_DOC_TYPES = ["standard_terms", "glossary"]
PAGE_DOC_TYPES = ["life_guide", "additional_p4l"]


@cache
def source_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    for t in PARENT_DOC_TYPES:
        for p in load_parents(MD_DIR / t, t):
            out[p.source_id] = p.content
    for t in PAGE_DOC_TYPES:
        for p in load_pages(MD_DIR / t, t):
            out[p.source_id] = p.content
    return out

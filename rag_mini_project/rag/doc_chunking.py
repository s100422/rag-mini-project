"""실험용 청킹 전략(docs/experiment_plan.md의 B0, E1~E4).

문서마다 본문 전체(DocText)를 만들고, 그 위에서 전략별로 청크를 자른다.
- recursive   : 문서 본문 전체를 Recursive로 자른다(B0, E1 크기, E2 겹침). 청크가 근거 경계를 넘을 수 있다.
- structure   : 구조 단위 안에서만 자른다(E3-a). 약관=조항, 용어집=용어, 가이드북=제목 구간, 추가 문서=쪽.
- parent-child: 구조 단위를 parent로, 그 안을 child로 자른다(E3-b). parent는 Postgres에 저장.
선택:
- title_prefix: 청크 앞에 `[문서 제목 > 상위 제목]`을 붙인다(E4-a, 임베딩·컨텍스트 모두).
- table_aware : 표는 행 경계에서만 자르고 조각마다 표 제목·머리행·단위·각주를 붙인다(E4-b).

청크 메타데이터의 source_ids에는 청크가 걸친 근거 ID(조항·용어·쪽)를 모두 넣는다.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .chunking import SEPARATORS, ChunkConfig
from .config import MD_DIR
from .md_parser import Parent, load_pages, load_parents

PARENT_DOC_TYPES = {"standard_terms", "glossary"}
STRATEGIES = ["recursive", "structure", "parent-child"]
# 쪽 단위 문서는 md에 문서 제목 줄이 없어 직접 적는다(1쪽 표지 기준)
DOC_TITLES = {
    "financial_life_guide": "20대에 꼭! 필요한 금융생활 가이드북",
    "housing_pension_2026": "2026 주택연금 백문백답",
    "housing_policy_finance_2025": "우리나라 주택 정책금융 현황과 평가",
    "insurance_terms_example_2020": "보험 약관 이용 가이드북 모범 예시",
    "policy_finance_debt_adjustment_2026": "포용적 금융 실현을 위한 정책서민금융 및 채무조정제도 분석",
    "retirement_pension_2025": "2025년 우리나라 퇴직연금 투자 백서",
    "self_employed_debt_2025": "최근 자영업자의 채무상환능력 평가 및 시사점",
}
MIN_SOURCE_OVERLAP = 10   # 청크가 근거와 이만큼(공백 뺀 글자) 겹쳐야 그 근거에 걸친 것으로 본다


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _clean_heading(h: str) -> str:
    return re.sub(r"\s+", " ", h.replace("**", "")).strip()


@dataclass
class Span:
    start: int
    end: int
    source_id: str


@dataclass
class DocText:
    doc_id: str
    doc_type: str
    source_file: str
    doc_title: str
    source_doc_id: str
    text: str = ""
    spans: list[Span] = field(default_factory=list)                 # 근거 단위 구간(조항·용어·쪽)
    pages: list[tuple[int, int]] = field(default_factory=list)      # (위치, 그 위치부터의 쪽)
    headings: list[tuple[int, str]] = field(default_factory=list)   # (위치, 그 위치부터의 상위 제목)
    units: list[tuple[int, int, str]] = field(default_factory=list)  # 구조 단위 (시작, 끝, 이름)
    parents: list[Parent] = field(default_factory=list)             # 약관·용어집의 원래 parent(E3-b에서 그대로 씀)

    def _append(self, content: str) -> int:
        if self.text:
            self.text += "\n\n"
        start = len(self.text)
        self.text += content
        return start

    def page_at(self, pos: int) -> int:
        i = bisect.bisect_right([p for p, _ in self.pages], pos) - 1
        return self.pages[max(i, 0)][1]

    def heading_at(self, pos: int) -> str:
        i = bisect.bisect_right([p for p, _ in self.headings], pos) - 1
        return self.headings[i][1] if i >= 0 else ""

    def source_ids(self, start: int, end: int) -> list[str]:
        out = []
        for s in self.spans:
            if s.end <= start or s.start >= end:
                continue
            overlap = _squash(self.text[max(s.start, start):min(s.end, end)])
            if len(overlap) >= min(MIN_SOURCE_OVERLAP, len(_squash(self.text[s.start:s.end]))):
                out.append(s.source_id)
        if not out:  # 공백만 겹친 경우: 가장 가까운 근거
            i = max(bisect.bisect_right([s.start for s in self.spans], start) - 1, 0)
            out = [self.spans[i].source_id]
        return out


# ---- 문서 본문 만들기 ----------------------------------------------------------

def load_docs(doc_type: str) -> list[DocText]:
    return _parent_docs(doc_type) if doc_type in PARENT_DOC_TYPES else _page_docs(doc_type)


def _parent_docs(doc_type: str) -> list[DocText]:
    """약관·용어집: parent(조항·용어)를 순서대로 이어 붙인다."""
    docs: dict[str, DocText] = {}
    for p in load_parents(MD_DIR / doc_type, doc_type):
        d = docs.setdefault(p.doc_id, DocText(p.doc_id, doc_type, p.source_file, p.doc_title, p.source_doc_id))
        start = d._append(p.content)
        end = len(d.text)
        d.spans.append(Span(start, end, p.source_id))
        d.pages.append((start, p.page_start))
        d.pages += [(start + pos, pg) for pos, pg in p.page_breaks]
        d.headings.append((start, _clean_heading(p.heading)))
        d.units.append((start, end, p.heading))
        d.parents.append(p)
    return list(docs.values())


def _page_docs(doc_type: str) -> list[DocText]:
    """가이드북·추가 문서: 쪽을 이어 붙인다. 구조 단위는 추가 문서=쪽, 가이드북=제목 구간."""
    docs: dict[str, DocText] = {}
    for pg in load_pages(MD_DIR / doc_type, doc_type):
        d = docs.setdefault(pg.doc_id, DocText(pg.doc_id, doc_type, pg.source_file,
                                               DOC_TITLES.get(pg.doc_id, pg.doc_id), pg.source_doc_id))
        start = d._append(pg.content)
        d.spans.append(Span(start, len(d.text), pg.source_id))
        d.pages.append((start, pg.page))
        cur = 0
        for h in pg.headings:
            i = pg.content.find(h, cur)
            if i >= 0:
                d.headings.append((start + i, _clean_heading(h)))
                cur = i + len(h)
    for d in docs.values():
        if doc_type == "life_guide":
            d.units = _heading_units(d)
        else:
            d.units = [(s.start, s.end, s.source_id) for s in d.spans]
    return list(docs.values())


def _heading_units(d: DocText) -> list[tuple[int, int, str]]:
    """제목 줄마다 끊는다. 제목만 있고 본문이 없는 구간은 다음 구간에 붙인다."""
    cuts = sorted({0, *[p for p, _ in d.headings]}) + [len(d.text)]
    units, pending = [], None
    for a, b in zip(cuts, cuts[1:]):
        start = a if pending is None else pending
        body = d.text[a:b].split("\n", 1)[1] if a in dict(d.headings) and "\n" in d.text[a:b] else d.text[a:b]
        if a in dict(d.headings) and not body.strip():
            pending = start
            continue
        units.append((start, b, d.heading_at(start) or "앞부분"))
        pending = None
    if pending is not None:
        units.append((pending, len(d.text), d.heading_at(pending)))
    return units


# ---- 자르기 -------------------------------------------------------------------

def _splitter(cfg: ChunkConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
                                          separators=SEPARATORS, keep_separator="end", add_start_index=True)


def _split(text: str, offset: int, cfg: ChunkConfig) -> list[tuple[int, int, str]]:
    if not text.strip():
        return []
    out = []
    for d in _splitter(cfg).create_documents([text]):
        s = offset + d.metadata["start_index"]
        out.append((s, s + len(d.page_content), d.page_content))
    return out


TABLE_LINE_RE = re.compile(r"^\s*\|")
CAPTION_MAX = 80   # 표 바로 앞 줄 중 이보다 짧은 것만 표 제목·단위로 본다(본문 문장 제외)
NOTE_RE = re.compile(r"^\s*(자료|출처|주\s*[:)]|주\d|\*|※|\d+\)|\(\d+\)|Note|note)")


def _table_runs(text: str) -> list[tuple[int, int]]:
    """표(| 로 시작하는 줄이 이어진 구간)의 (시작, 끝) 위치."""
    runs, start, pos, prev_end = [], None, 0, 0
    for line in text.splitlines(keepends=True):
        if TABLE_LINE_RE.match(line):
            if start is None:
                start = pos
            prev_end = pos + len(line.rstrip("\n"))
        elif line.strip() and start is not None:
            runs.append((start, prev_end))
            start = None
        pos += len(line)
    if start is not None:
        runs.append((start, prev_end))
    return runs


def _table_chunks(text: str, a: int, b: int, cfg: ChunkConfig) -> list[tuple[int, int, str]]:
    """표 하나를 행 경계에서 자르고, 조각마다 표 제목·단위(앞 2줄)·머리행·각주(뒤 3줄)를 붙인다."""
    before = [l for l in text[max(0, a - 300):a].splitlines() if l.strip()][-2:]
    caption = [l for l in before if len(l) <= CAPTION_MAX and not TABLE_LINE_RE.match(l)]
    after = [l for l in text[b:b + 400].splitlines() if l.strip()][:3]
    notes = []
    for l in after:
        if not NOTE_RE.match(l):
            break
        notes.append(l)
    lines = text[a:b].split("\n")
    is_sep = len(lines) > 1 and re.fullmatch(r"[\s|:\-]+", lines[1])
    header = lines[:2] if is_sep else lines[:1]
    rows = lines[len(header):]
    # 줄 위치(문서 안)
    row_pos, p = [], a + sum(len(l) + 1 for l in header)
    for l in rows:
        row_pos.append((p, p + len(l)))
        p += len(l) + 1

    def compose(rs: list[str]) -> str:
        return "\n".join([*caption, *header, *rs, *notes])

    if len(compose(rows)) <= cfg.chunk_size or not rows:
        return [(a, b, compose(rows))]
    # 한 행이 혼자서도 넘치면 셀 경계(|)에서 다시 나눈다(조각마다 머리행이 붙는다)
    budget = max(cfg.chunk_size - len(compose([])) - 1, 100)
    cell_splitter = RecursiveCharacterTextSplitter(chunk_size=budget, chunk_overlap=0,
                                                   separators=[" | ", " ", ""], keep_separator="start")
    split_rows = []
    for pos, r in zip(row_pos, rows):
        pieces = [r] if len(r) <= budget else cell_splitter.split_text(r)
        split_rows += [(pos, p) for p in pieces]
    row_pos, rows = [p for p, _ in split_rows], [r for _, r in split_rows]
    out, cur, cur_start = [], [], None
    for (s, e), r in zip(row_pos, rows):
        if cur and len(compose(cur + [r])) > cfg.chunk_size:
            out.append((cur_start, cur_end, compose(cur)))
            cur, cur_start = [], None
        if cur_start is None:
            cur_start = s if out else a  # 첫 조각은 머리행부터
        cur.append(r)
        cur_end = e
    if cur:
        out.append((cur_start, cur_end, compose(cur)))
    return out


def _segments_table_aware(d: DocText, a: int, b: int, cfg: ChunkConfig) -> list[tuple[int, int, str]]:
    """구간 [a, b)를 표와 표 밖으로 나눠, 표 밖은 Recursive로, 표는 _table_chunks로 자른다."""
    out, pos = [], a
    for ta, tb in _table_runs(d.text[a:b]):
        ta, tb = a + ta, a + tb
        out += _split(d.text[pos:ta], pos, cfg)
        out += _table_chunks(d.text, ta, tb, cfg)
        pos = tb
    out += _split(d.text[pos:b], pos, cfg)
    return out


def chunk_docs(docs: list[DocText], strategy: str, cfg: ChunkConfig, title_prefix: bool = False,
               table_aware: bool = False) -> list[Document]:
    """recursive·structure 전략의 청크(parent 없음)."""
    if strategy not in ("recursive", "structure"):
        raise ValueError(strategy)
    tag = tag_of(strategy, cfg, title_prefix, table_aware)
    out: list[Document] = []
    for d in docs:
        regions = [(0, len(d.text))] if strategy == "recursive" else [(s, e) for s, e, _ in d.units]
        pieces = []
        for a, b in regions:
            pieces += (_segments_table_aware(d, a, b, cfg) if table_aware
                       else _split(d.text[a:b], a, cfg))
        for i, (s, e, text) in enumerate(pieces):
            out.append(_document(d, f"{d.doc_id}::{tag}::{i:04d}", s, e, text, i, len(pieces),
                                 strategy, cfg, title_prefix))
    return out


def _document(d: DocText, chunk_id: str, s: int, e: int, text: str, i: int, n: int, strategy: str,
              cfg: ChunkConfig, title_prefix: bool, parent_id: str | None = None) -> Document:
    heading = d.heading_at(s)
    if title_prefix:
        text = f"[{d.doc_title} > {heading}]\n{text}" if heading else f"[{d.doc_title}]\n{text}"
    sids = d.source_ids(s, e)
    meta = {
        "chunk_id": chunk_id, "doc_id": d.doc_id, "doc_type": d.doc_type, "source_file": d.source_file,
        "doc_title": d.doc_title, "heading": heading, "chunk_index": i, "chunk_count": n, "start_index": s,
        "page_start": d.page_at(s), "page_end": d.page_at(max(s, e - 1)),
        "source_id": sids[0], "source_ids": sids, "source_doc_id": d.source_doc_id,
        "chunk_config": cfg.name, "chunk_strategy": strategy, "parent_id": parent_id,
    }
    return Document(page_content=text, metadata={k: v for k, v in meta.items() if v is not None})


# ---- parent-child (E3-b) -----------------------------------------------------

def unit_parents(docs: list[DocText]) -> list[Parent]:
    """구조 단위를 parent로. 약관·용어집은 원래 parent 그대로, 쪽 단위 문서는 단위마다 새로 만든다."""
    out: list[Parent] = []
    for d in docs:
        if d.parents:
            out += d.parents
            continue
        for i, (s, e, name) in enumerate(d.units):
            sids = d.source_ids(s, e)
            breaks = sorted({(pos - s, pg) for pos, pg in d.pages if s < pos < e})
            out.append(Parent(
                parent_id=f"{d.doc_id}::u{i:03d}", doc_id=d.doc_id, doc_type=d.doc_type,
                source_file=d.source_file, doc_title=d.doc_title, seq=i, heading=_clean_heading(name),
                article_no=None, article_title=None, content=d.text[s:e],
                page_start=d.page_at(s), page_end=d.page_at(max(s, e - 1)), page_breaks=breaks,
                source_id=sids[0], source_doc_id=d.source_doc_id, source_ids=sids,
            ))
    return out


def tag_of(strategy: str, cfg: ChunkConfig, title_prefix: bool, table_aware: bool) -> str:
    short = {"recursive": "rec", "structure": "struct", "parent-child": "pc"}[strategy]
    return "_".join([short, *(["title"] if title_prefix else []), *(["table"] if table_aware else []), cfg.name])

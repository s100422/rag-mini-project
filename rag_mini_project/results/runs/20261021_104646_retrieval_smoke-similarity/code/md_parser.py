"""pymupdf4llm으로 만든 md를 읽어 `##` 제목 단위의 parent로 나눈다.

md 형식(extract/extract_terms.py 출력):
    <!-- source: 04_신용거래약관.pdf -->
    <!-- page: 1 -->
    # 신용거래약관                      ← 문서 제목
    금융투자협회 | 원문 기준일: … | 버전: …
    ## 제1조(약관의 적용)               ← parent 시작
    …
    # 출처 목록                         ← 조항별 원문 URL (parent에서 빼고 메타데이터로)

정리 규칙
- `<!-- page: N -->`는 본문에서 빼고, parent 안 위치(page_breaks)만 기록해 child 쪽 번호를 계산할 때 쓴다.
- 숫자만 있는 줄(쪽 번호)은 버린다.
- 본문 안의 `###` 이하 제목은 pymupdf4llm이 본문 줄을 잘못 제목으로 잡은 것이라 `#`을 떼고 본문으로 둔다.

용어집(glossary)도 같은 형식이다: `## 용어` 하나가 parent, 출처 목록은 '용어 / 용어 설명 / URL(dicId)'.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

PAGE_RE = re.compile(r"^<!--\s*page:\s*(\d+)\b[^>]*-->$")  # 뒤에 "(printed 82)"가 붙기도 한다
SOURCE_RE = re.compile(r"^<!--\s*source:\s*(.+?)\s*-->$")
ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\((.*)\))?")
DOC_META_RE = re.compile(r"원문 기준일:\s*([^|]+?)\s*\|\s*버전:\s*(\S+)")
SOURCE_LIST_TITLE = "출처 목록"


@dataclass
class Parent:
    parent_id: str
    doc_id: str
    doc_type: str
    source_file: str
    doc_title: str
    seq: int                      # 문서 안 순서(0부터)
    heading: str                  # 예: 제1조(약관의 적용)
    article_no: str | None        # 예: 제1조, 제2조의2
    article_title: str | None     # 예: 약관의 적용
    content: str                  # 제목 줄을 포함한 parent 원문
    page_start: int
    page_end: int
    page_breaks: list[tuple[int, int]] = field(default_factory=list)  # (content 안 위치, 그 위치부터의 쪽)
    source_url: str | None = None
    effective_date: str | None = None
    version: str | None = None
    # 골든셋(03_answers)의 expected_source_ids와 같은 형식
    source_id: str | None = None      # 조항 단위: kofia-terms-123-1709-015
    source_doc_id: str | None = None  # 문서 단위: kofia-terms-123-1709

    def page_at(self, offset: int) -> int:
        page = self.page_start
        for pos, p in self.page_breaks:
            if pos <= offset:
                page = p
            else:
                break
        return page


SENTENCE_END_RE = re.compile(r"(다|함|음|임|것|등)\s*\.?\s*$|[.:)\]>]\s*$")
ITEM_START_RE = re.compile(r"^(?:[-*]\s+)?(?:[①-⑳]|\d+(?:의\d+)?\.|[가-하]\.|\(\d+\)|<|\[|\|)")


def _continues(prev: str, nxt: str) -> bool:
    """앞 문단이 문장 끝이 아니고 다음 문단이 항·호 번호로 시작하지 않으면 이어지는 문장으로 본다."""
    return not SENTENCE_END_RE.search(prev) and not ITEM_START_RE.match(nxt)


def _article(heading: str) -> tuple[str | None, str | None]:
    m = ARTICLE_RE.match(heading)
    if not m:
        return None, None
    no = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
    return no, (m.group(3) or "").strip() or None


class _Builder:
    def __init__(self, heading: str, page: int):
        self.heading = heading
        self.page_start = page
        self.page_end = page
        self.paras: list[str] = [heading]
        self.breaks: list[tuple[int, int]] = []
        self.pending_page: int | None = None

    def _len(self) -> int:
        return len("\n\n".join(self.paras))

    def add(self, text: str) -> None:
        if self.pending_page is not None:
            self.page_end = self.pending_page
            # 쪽 경계에서 끊긴 문장은 앞 문단에 이어 붙인다(줄 잇기와 같게 공백 하나)
            if len(self.paras) > 1 and _continues(self.paras[-1], text):
                self.breaks.append((self._len() + 1, self.pending_page))
                self.paras[-1] += " " + text
                self.pending_page = None
                return
            self.breaks.append((self._len() + 2, self.pending_page))
            self.pending_page = None
        self.paras.append(text)

    def page(self, p: int) -> None:
        # 쪽 표시 뒤에 본문이 이어질 때만 page_end를 늘린다(다음 제목 바로 앞의 쪽 표시는 넘긴다)
        self.pending_page = p

    def content(self) -> str:
        return "\n\n".join(self.paras)


SOURCE_ENTRY_RE = re.compile(
    r"(제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*(?:\([^)]*\))?\s*/\s*(h\s*t\s*t\s*p.*?)(?=/\s*제\s*\d+\s*조|$)")
URL_RE = re.compile(r"https?://[\x21-\x7e]+")


def _parse_source_list(text: str) -> dict[str, str]:
    """'약관명 / 제1조(…) / https://… 약관명 / 제2조(…) / …' → {조항: URL}.

    한 줄에 여러 항목이 이어지거나 URL 중간에 공백이 끼어 있어(추출 잡음),
    URL 부분의 공백을 모두 지운 뒤 ASCII 구간만 URL로 본다(뒤따르는 약관명은 한글이라 잘린다).
    """
    urls: dict[str, str] = {}
    for m in SOURCE_ENTRY_RE.finditer(text):
        no, _ = _article(m.group(1))
        url = URL_RE.match(re.sub(r"\s+", "", m.group(2)))
        if no and url:
            urls[no] = url.group(0)
    return urls


KOFIA_URL_RE = re.compile(r"[?&]seq=(\d+)&historySeq=(\d+)&contentSeq=(\d+)")


def _assign_source_ids(parents: list["Parent"]) -> None:
    """출처 URL로 골든셋 근거 ID를 붙인다.

    kofia-terms-{seq}-{historySeq}-{NNN}에서 NNN은 조항 번호가 아니라 문서 안 조항 순서(1부터)다.
    조의N 조항이 있는 문서에서 둘이 갈라진다(예: 파생상품약관 14번째 조항 = 제12조).
    contentSeq가 조항 순서대로 1씩 늘어나므로 첫 조항과의 차이로 순서를 구한다.
    """
    parsed = [(p, KOFIA_URL_RE.search(p.source_url or "")) for p in parents]
    seqs = [int(m[3]) for _, m in parsed if m]
    if not seqs:
        return
    first = min(seqs)
    for p, m in parsed:
        if not m:
            continue
        ordinal = int(m[3]) - first + 1
        if ordinal != p.seq + 1:
            raise ValueError(f"{p.parent_id}: 출처 URL 순서({ordinal})와 md 조항 순서({p.seq + 1})가 다릅니다")
        p.source_doc_id = f"kofia-terms-{m[1]}-{m[2]}"
        p.source_id = f"{p.source_doc_id}-{ordinal:03d}"


GLOSSARY_ENTRY_RE = re.compile(r"(.+?)\s*/\s*용어 설명\s*/\s*(h\s*t\s*t\s*p.*?dicId\s*=\s*(\d+))")


def _assign_glossary_ids(parents: list["Parent"], source_text: str) -> None:
    """용어집: '용어 / 용어 설명 / https://…?dicId=1697&curPage=23 …' → fsc-glossary-{dicId}.

    같은 이름의 용어가 dicId만 달리해 여러 번 나오므로(정리제도 3개 등) 이름이 아니라 순서로 짝짓고,
    이름이 맞는지 확인한다. 항목 이름 앞에는 앞 URL의 꼬리(추출 잡음)가 붙을 수 있어 끝부분만 비교한다.
    parent_id도 dicId로 만든다(용어 이름은 겹친다).
    """
    entries = GLOSSARY_ENTRY_RE.findall(source_text)
    if len(entries) != len(parents):
        raise ValueError(f"용어집 출처 항목 수({len(entries)})와 용어 수({len(parents)})가 다릅니다")
    for p, (name, url, dic_id) in zip(parents, entries):
        if not re.sub(r"\s+", "", name).endswith(re.sub(r"\s+", "", p.heading)):
            raise ValueError(f"{p.heading}: 출처 목록 순서가 맞지 않습니다 ({name!r})")
        p.parent_id = f"{p.doc_id}::{dic_id}"
        p.source_url = re.sub(r"\s+", "", url)
        p.source_doc_id = "fsc-glossary"
        p.source_id = f"fsc-glossary-{dic_id}"


def parse_md(path: Path, doc_type: str) -> list[Parent]:
    doc_id = path.stem
    source_file = doc_id + ".pdf"
    doc_title = doc_id
    effective_date = version = None
    page = 1
    cur: _Builder | None = None
    builders: list[_Builder] = []
    in_sources = False
    source_lines: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := SOURCE_RE.match(line):
            source_file = m.group(1)
            continue
        if m := PAGE_RE.match(line):
            page = int(m.group(1))
            if cur and not in_sources:
                cur.page(page)
            continue
        if line.isdigit():  # 쪽 번호 줄
            continue

        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            if title == SOURCE_LIST_TITLE:
                in_sources = True
                cur = None
            elif not builders and cur is None:
                doc_title = title
            continue

        if in_sources:
            source_lines.append(line)
            continue

        if line.startswith("## "):
            cur = _Builder(line[3:].strip(), page)
            builders.append(cur)
            continue

        if cur is None:  # 첫 조항 앞(제목 아래 기관·기준일 줄)
            if m := DOC_META_RE.search(line):
                effective_date, version = m.group(1).strip(), m.group(2)
            continue

        cur.add(re.sub(r"^#{3,6}\s+", "", line))

    source_urls = _parse_source_list(" ".join(source_lines))
    parents = []
    for i, b in enumerate(builders):
        no, title = _article(b.heading) if doc_type != "glossary" else (None, None)
        parents.append(Parent(
            parent_id=f"{doc_id}::{no or f's{i:03d}'}",
            doc_id=doc_id,
            doc_type=doc_type,
            source_file=source_file,
            doc_title=doc_title,
            seq=i,
            heading=b.heading,
            article_no=no,
            article_title=title,
            content=b.content(),
            page_start=b.page_start,
            page_end=b.page_end,
            page_breaks=b.breaks,
            source_url=source_urls.get(no) if no else None,
            effective_date=effective_date,
            version=version,
        ))
    if doc_type == "glossary":
        _assign_glossary_ids(parents, " ".join(source_lines))
    else:
        _assign_source_ids(parents)
    ids = [p.parent_id for p in parents]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: parent_id가 겹칩니다")
    return parents


def load_parents(md_dir: Path, doc_type: str) -> list[Parent]:
    out: list[Parent] = []
    for path in sorted(md_dir.glob("*.md")):
        out.extend(parse_md(path, doc_type))
    return out


# ---- 쪽 단위 문서(가이드북·추가 문서): parent 없이 쪽별 본문만 만든다 ---------------------------
#
# md 형식(extract/extract_p4l.py 출력):
#     <!-- page: 9 -->
#     <!-- block: graph id=p9-g1 page=9 source=vision … title="연령별 금융이해력 점수" -->
#     …블록 본문…
#     <!-- /block -->
# 주석 줄은 버리고 본문 줄은 모두 남긴다. 그림·표 제목(title)이 본문에 없으면 블록 앞에 붙인다.
# `#` 제목 표시는 레벨이 들쭉날쭉해 떼고 본문으로 둔다.

BLOCK_OPEN_RE = re.compile(r"^<!--\s*block:.*?-->$")
BLOCK_CLOSE_RE = re.compile(r"^<!--\s*/block\s*-->$")
BLOCK_TITLE_RE = re.compile(r'title="([^"]*)"')
COMMENT_RE = re.compile(r"^<!--.*-->$")
HEADING_RE = re.compile(r"^#{1,6}\s+")

# 골든셋 근거 ID 앞부분(기본은 doc_id): fss-life-guide-pdf-25, housing_pension_2026-pdf-15
SOURCE_DOC_IDS = {"financial_life_guide": "fss-life-guide"}


@dataclass
class PageText:
    doc_id: str
    doc_type: str
    source_file: str
    page: int
    content: str
    source_id: str
    source_doc_id: str


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def parse_pages(path: Path, doc_type: str) -> list[PageText]:
    doc_id = path.stem
    source_file = doc_id + ".pdf"
    source_doc_id = SOURCE_DOC_IDS.get(doc_id, doc_id)
    pages: dict[int, list[str]] = {}
    page = 1
    block: list[str] | None = None
    title = ""

    def emit(lines: list[str]) -> None:
        text = "\n".join(lines).strip()
        if text:
            pages.setdefault(page, []).append(text)

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if m := SOURCE_RE.match(line):
            source_file = m.group(1)
            continue
        if m := PAGE_RE.match(line):
            page = int(m.group(1))
            continue
        if BLOCK_OPEN_RE.match(line):
            block = []
            t = BLOCK_TITLE_RE.search(line)
            title = t.group(1).strip() if t else ""
            continue
        if BLOCK_CLOSE_RE.match(line):
            body = block or []
            if title and _squash(title) not in _squash("\n".join(body)):
                body = [title, *body]
            emit(body)
            block, title = None, ""
            continue
        if COMMENT_RE.match(line):
            continue
        text = HEADING_RE.sub("", raw.rstrip())
        if block is not None:
            block.append(text)
        else:
            emit([text])

    out = []
    for p, parts in sorted(pages.items()):
        content = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts))
        out.append(PageText(doc_id, doc_type, source_file, p, content,
                            f"{source_doc_id}-pdf-{p}", source_doc_id))
    return out


def load_pages(md_dir: Path, doc_type: str) -> list[PageText]:
    out: list[PageText] = []
    for path in sorted(md_dir.glob("*.md")):
        out.extend(parse_pages(path, doc_type))
    return out

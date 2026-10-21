"""pdfplumber로 PDF를 Markdown으로 추출한다(보도자료·보고서형 1단 문서용).

처리 순서
0. 문서 전체 줄을 먼저 읽어 줄 잇기 증거(붙인 형태·띄운 형태 빈도)를 모은다.
1. 표(쪽마다): find_tables로 찾는다.
   - 안의 글자가 15pt 이상인 짧은 상자는 장 제목 상자(Ⅰ. …)로 보고 제목으로 쓴다.
   - 진짜 표는 extract/tables.py로 병합 칸을 채우고 머리글을 한 줄로 합친다.
     검사(열 이름 중복·빠진 글자 등)에 걸리면 표 이미지를 Gemini 비전으로 다시 읽는다.
   - 표 위 제목(굵은 13pt)·단위 줄과 표 아래 주석(*, ※, 주:)은 표 블록에 함께 넣는다.
2. 그림: 래스터 이미지(로고·아이콘 같은 작은 것은 제외)를 붙어 있는 것끼리 한 영역으로 묶어
   잘라내고 Gemini 비전으로 읽는다. 영역 안 글자 줄은 본문에서 뺀다(비전 결과에 들어감).
3. 줄: extract_text_lines로 줄을 얻고, 표·그림과 함께 위→아래 순서로 놓는다.
4. 문단: □ ◦ * ※ 등으로 시작하는 줄은 새 문단, 나머지는 앞 줄에 잇는다.
   한글 문서는 단어 중간에서 줄이 바뀌므로(비롯/하여) 이을 때 띄울지 붙일지를 정한다:
   문서 안에 붙인 형태가 더 많이 나오면 붙이고, 띄운 형태가 더 많으면 띄우고, 둘 다 없으면 Kiwi에 맡긴다.
5. 제목: 첫 쪽 18pt 이상 → #, 장(Ⅰ·I.) → ##, 절(1 …, 15pt 이상) → ###, 표 제목 → **굵게**.

출력 표시(청킹·검색·답변 단계에서 씀)
  <!-- page: N -->
  <!-- block: table id=p4-t1 page=4 source=rule|vision [image="…png"] --> … <!-- /block -->
  <!-- block: figure id=p27-f1 page=27 source=vision image="…png" --> … <!-- /block -->

사용법: python extract/extract_pdfplumber.py 01_data/additional/retirement_pension_2025.pdf [-o md/additional] [--no-vision]
"""
import argparse
import io
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from kiwipiepy import Kiwi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tables import extract_table  # noqa: E402
import vision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# 문단을 새로 시작하는 머리 기호. U+E000~F8FF(사용자 정의 영역)는 한글 문서의 네모·화살표 기호다.
BULLET_RE = re.compile(r"^\s*(□|■|◦|○|●|⦁|▪|‣|•|-|\*|※|①|②|③|④|⑤|\d+\)|<|\[|ㅇ|[-])")
NOTE_RE = re.compile(r"^\s*(\*|※|주\s*[:：)]|주\d|자료\s*[:：])")
UNIT_RE = re.compile(r"^\(\s*단위")
CHAPTER_RE = re.compile(r"^([IVX]+\.?|[Ⅰ-Ⅻ]\.?)(\s|$)")
SECTION_RE = re.compile(r"^\d{1,2}\s+\S")
PAGE_NO_RE = re.compile(r"^[-–]\s*\d+\s*[-–]$|^\d+$")
TOC_LEADER_RE = re.compile(r"[·…\.]{6,}")
UNIT_AFTER_NUMBER_RE = re.compile(r"^(년|월|일|원|조|억|만|천|%|개|명|좌|배|세|분기|차|건|회)")
TITLE_BOX_SIZE = 15      # 이보다 큰 글자가 든 짧은 표 상자는 장 제목
DOC_TITLE_SIZE = 18
MIN_SIZE = 7             # 이보다 작은 글자 줄(장식)은 버림
MIN_FIG_SIDE = 40        # 그림으로 볼 최소 크기(pt)와 쪽 면적 비율 — 로고·아이콘 제외
MIN_FIG_AREA = 0.01
FIG_MERGE_GAP = 12       # 이 간격 이하로 붙은 이미지는 한 그림으로 묶음
RENDER_DPI = 200


# ---- 줄 잇기 --------------------------------------------------------------

class Joiner:
    """줄바꿈 자리에서 앞 줄 끝 단어와 다음 줄 첫 단어를 붙일지 정한다."""

    def __init__(self, lines: list[str]):
        # 증거는 한 줄 안에서만 센다(줄바꿈 자리 자체를 '띄운 형태'로 세지 않도록)
        self.unigram: Counter = Counter()
        self.bigram: Counter = Counter()
        for line in lines:
            tokens = line.split()
            self.unigram.update(tokens)
            self.bigram.update(zip(tokens, tokens[1:]))
        self.kiwi = Kiwi()

    def join(self, prev: str, nxt: str) -> str:
        if not prev:
            return nxt
        a, b = prev.split()[-1], nxt.split()[0]
        if re.search(r"\d$", a) and UNIT_AFTER_NUMBER_RE.match(b):
            return prev + nxt  # 1995 / 년 → 1995년
        if not (re.search(r"[가-힣]$", a) and re.match(r"[가-힣]", b)):
            return prev + " " + nxt
        glued, spaced = self.unigram[a + b], self.bigram[(a, b)]
        if glued != spaced:
            return prev + ("" if glued > spaced else " ") + nxt
        # 문서에 증거가 없으면 Kiwi에 앞뒤 단어만 보여 주고 판단을 맡긴다
        sep = "" if " " not in self.kiwi.space(a + b).strip() else " "
        return prev + sep + nxt


# ---- 블록 ------------------------------------------------------------------

@dataclass
class Block:
    top: float
    bottom: float
    kind: str                 # heading1·heading2·heading3·caption·line·table·figure
    text: str = ""
    line: dict | None = None  # kind == line 일 때 원래 줄
    # 표·그림
    block_id: str = ""
    source: str = "rule"
    image: str = ""           # md 파일 기준 상대 경로
    png: bytes = b""          # 비전으로 읽을 이미지
    vision_kind: str = ""     # figure | table
    caption: str = ""
    unit: str = ""
    notes: list[str] | None = None
    problems: list[str] | None = None


def _inside(obj: dict, bbox) -> bool:
    x0, top, x1, bottom = bbox
    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _render(page, bbox) -> bytes:
    x0, top, x1, bottom = bbox
    box = (max(0, x0 - 4), max(0, top - 4), min(page.width, x1 + 4), min(page.height, bottom + 4))
    img = page.crop(box).to_image(resolution=RENDER_DPI).original
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def figure_regions(page) -> list[tuple]:
    """의미 있는 래스터 이미지를 붙어 있는 것끼리 묶은 영역."""
    area = page.width * page.height
    boxes = []
    for im in page.images:
        x0, top, x1, bottom = max(0, im["x0"]), max(0, im["top"]), min(page.width, im["x1"]), min(page.height, im["bottom"])
        w, h = x1 - x0, bottom - top
        if w >= MIN_FIG_SIDE and h >= MIN_FIG_SIDE and w * h / area >= MIN_FIG_AREA:
            boxes.append([x0, top, x1, bottom])
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a[0] - FIG_MERGE_GAP <= b[2] and b[0] - FIG_MERGE_GAP <= a[2]
                        and a[1] - FIG_MERGE_GAP <= b[3] and b[1] - FIG_MERGE_GAP <= a[3]):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return sorted((tuple(b) for b in boxes), key=lambda b: (b[1], b[0]))


# ---- 쪽 처리 ----------------------------------------------------------------

def page_blocks(page, page_no: int, joiner: Joiner, img_dir: Path, img_rel: str, use_vision: bool) -> list[Block]:
    blocks: list[Block] = []
    tables, boxes = [], []

    for t in page.find_tables():
        chars = [c for c in page.chars if _inside(c, t.bbox)]
        if not chars:
            continue
        if max(c["size"] for c in chars) >= TITLE_BOX_SIZE:
            # 짧은 큰 글씨 상자만 제목으로, 쪽 전체를 감싼 상자(안내 그림 등)는 본문 줄로 둔다
            if len(chars) <= 80:
                boxes.append(t.bbox)
            continue
        res = extract_table(page, t, joiner.join)
        if not res.rows or sum(1 for r in res.rows for c in r if c) + len(res.header) < 4:
            continue
        tables.append(t.bbox)
        b = Block(t.bbox[1], t.bbox[3], "table", res.to_md(), block_id=f"p{page_no}-t{len(tables)}",
                  problems=res.problems)
        if res.problems and use_vision:
            b.png, b.vision_kind, b.source = _render(page, t.bbox), "table", "vision"
            b.image = f"{img_rel}/{b.block_id}.png"
            (img_dir / f"{b.block_id}.png").write_bytes(b.png)
        blocks.append(b)

    figures = figure_regions(page) if use_vision else []
    for i, bbox in enumerate(figures, start=1):
        fid = f"p{page_no}-f{i}"
        png = _render(page, bbox)
        (img_dir / f"{fid}.png").write_bytes(png)
        blocks.append(Block(bbox[1], bbox[3], "figure", block_id=fid, source="vision",
                            image=f"{img_rel}/{fid}.png", png=png, vision_kind="figure"))

    box_lines: dict[int, list] = {}
    for line in page.extract_text_lines():
        size = max(c["size"] for c in line["chars"])
        text = line["text"].strip()
        if not text or size < MIN_SIZE or PAGE_NO_RE.match(text) or TOC_LEADER_RE.search(text):
            continue  # 빈 줄, 장식, 쪽 번호, 목차(점선 리더) 줄
        if any(_inside(line, b) for b in tables) or any(_inside(line, b) for b in figures):
            continue
        box = next((i for i, b in enumerate(boxes) if _inside(line, b)), None)
        if box is not None:
            box_lines.setdefault(box, []).append(line)
            continue
        bold = all("Bold" in c["fontname"] for c in line["chars"] if c["text"].strip())
        if page_no == 1 and size >= DOC_TITLE_SIZE:
            kind = "heading1"
        elif size >= TITLE_BOX_SIZE - 0.5 and CHAPTER_RE.match(text):
            kind = "heading2"
        elif size >= TITLE_BOX_SIZE - 0.5 and SECTION_RE.match(text) and not BULLET_RE.match(text):
            kind = "heading3"
        elif bold and 12.5 <= size < 14 and not BULLET_RE.match(text):
            kind = "caption"
        else:
            kind = "line"
        blocks.append(Block(line["top"], line["bottom"], kind, text, line=line))

    # 제목 상자: 장 번호(Ⅲ)와 제목을 한 줄로
    for i, lines in box_lines.items():
        lines.sort(key=lambda l: (round(l["top"] / 5), l["x0"]))
        blocks.append(Block(boxes[i][1], boxes[i][3], "heading2", " ".join(l["text"].strip() for l in lines)))

    blocks.sort(key=lambda b: b.top)
    blocks = _merge_chapter_subtitles(blocks)
    return _attach_table_context(blocks)


def _merge_chapter_subtitles(blocks: list[Block]) -> list[Block]:
    """장 제목 바로 아래 큰 글씨 줄(부제)은 장 제목에 붙인다: 'III 연금부자론' + '평범한 김과장이 …'"""
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (prev and prev.kind == "heading2" and b.kind == "line" and b.top - prev.top < 40
                and max(c["size"] for c in b.line["chars"]) >= TITLE_BOX_SIZE - 0.5 and not BULLET_RE.match(b.text)):
            prev.text = f"{prev.text} {b.text}"
            continue
        out.append(b)
    return out


def _attach_table_context(blocks: list[Block]) -> list[Block]:
    """표 바로 위 제목·단위 줄과 바로 아래 주석 줄을 표 블록에 넣는다."""
    drop: set[int] = set()
    for i, b in enumerate(blocks):
        if b.kind != "table":
            continue
        j = i - 1
        while j >= 0 and j not in drop and b.top - blocks[j].bottom < 60:
            p = blocks[j]
            if p.kind == "line" and UNIT_RE.match(p.text) and not b.unit:
                b.unit = p.text
            elif p.kind == "caption" and not b.caption:
                b.caption = p.text
            else:
                break
            drop.add(j)
            j -= 1
        notes, j, last = [], i + 1, b.bottom
        while j < len(blocks) and blocks[j].kind == "line" and NOTE_RE.match(blocks[j].text) and blocks[j].top - last < 25:
            notes.append(blocks[j].text)
            last = blocks[j].bottom
            drop.add(j)
            j += 1
        b.notes = notes
    return [b for i, b in enumerate(blocks) if i not in drop]


# ---- 문서 ------------------------------------------------------------------

def to_markdown(pdf_path: Path, out_dir: Path, use_vision: bool = True) -> tuple[str, dict]:
    img_rel = f"images/{pdf_path.stem}"
    img_dir = out_dir / img_rel
    img_dir.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        joiner = Joiner([l["text"] for p in pdf.pages for l in p.extract_text_lines()])
        pages = [(i + 1, page_blocks(p, i + 1, joiner, img_dir, img_rel, use_vision)) for i, p in enumerate(pdf.pages)]

    # 비전 대상은 한꺼번에 동시 호출(캐시 있으면 호출 안 함)
    jobs = [b for _, bl in pages for b in bl if b.png]
    if jobs:
        print(f"vision: {len(jobs)} images ({vision.MODEL})", flush=True)
        for b, text in zip(jobs, vision.describe_many([(b.png, b.vision_kind) for b in jobs])):
            b.text = text

    stats = Counter()
    out = [f"<!-- source: {pdf_path.name} -->"]
    for page_no, blocks in pages:
        out.append(f"<!-- page: {page_no} -->")
        para, prev_line = "", None
        for b in blocks:
            if b.kind == "line":
                gap = b.top - prev_line.bottom if prev_line else 999
                size = max(c["size"] for c in b.line["chars"])
                if para and not BULLET_RE.match(b.text) and gap < size * 0.9:
                    para = joiner.join(para, b.text)
                else:
                    if para:
                        out.append(para)
                    para = b.text
                prev_line = b
                continue
            if para:
                out.append(para)
                para, prev_line = "", None
            if b.kind in ("table", "figure"):
                if b.kind == "figure" and b.text.strip() in ("", "내용 없음"):
                    stats["figure_empty"] += 1
                    continue
                stats[f"{b.kind}_{b.source}"] += 1
                out.append(_block_md(b, page_no))
                continue
            prefix = {"heading1": "# ", "heading2": "## ", "heading3": "### "}.get(b.kind)
            out.append(prefix + b.text if prefix else f"**{b.text}**" if b.kind == "caption" else b.text)
        if para:
            out.append(para)
    return "\n\n".join(out) + "\n", dict(stats)


def _block_md(b: Block, page_no: int) -> str:
    attrs = f'{b.kind} id={b.block_id} page={page_no} source={b.source}'
    if b.image:
        attrs += f' image="{b.image}"'
    parts = [f"<!-- block: {attrs} -->"]
    if b.caption:
        parts.append(f"**{b.caption}**")
    if b.unit:
        parts.append(b.unit)
    parts.append(b.text.strip())
    parts += b.notes or []
    parts.append("<!-- /block -->")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, default=ROOT / "md" / "additional")
    ap.add_argument("--no-vision", action="store_true", help="비전 호출 없이 규칙만으로 추출")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md, stats = to_markdown(args.pdf, args.out_dir, use_vision=not args.no_vision)
    out = args.out_dir / (args.pdf.stem + ".md")
    out.write_text(md, encoding="utf-8")
    print(f"blocks: {stats}")
    print(f"done -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

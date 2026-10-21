"""pymupdf4llm + Gemini 비전으로 PDF를 Markdown으로 추출한다(쪽 단위).

쪽마다
[1] pymupdf4llm.to_markdown(page_chunks=True, use_ocr=NEVER, force_text=False)
    → 본문·표를 쪽별 Markdown으로. 레이아웃 상자(page_boxes)마다 종류와 본문 안 위치가 함께 나온다.
    - force_text=False: 그림 영역 안 글자를 본문에서 뺀다(그림은 [3]에서 비전으로 읽음).
      단, 그림으로 잡힌 상자에 한글 문장이 많으면(설명 박스) 글자층 문장을 본문으로 살린다.
      (ignore_images/ignore_graphics는 2.0 레이아웃 모드에서 지원하지 않아 쓰지 않는다)
    - table 상자 자리는 자체 표 로더(extract/tables.py)로 바꾼다:
      PyMuPDF find_tables의 칸 좌표로 세로 병합은 나눠 채우고, 가로 병합은 머리글을 합친다('은행 금액').
      검사에 걸리면 표 영역을 비전으로 다시 읽고, 맞는 칸 좌표가 없으면 pymupdf4llm 표를 그대로 둔다.
    - 칸 색칠로 범위를 나타낸 표(간트형)는 색 사각형 좌표로 읽어 범위 표로 바꾼다('30년형 | 55~57', source=vector).
    - 2단 쪽은 레이아웃 상자를 '왼쪽 단 → 오른쪽 단' 순서로 다시 놓는다(넓은 상자·각주는 구분선).
    - 그림으로 잡힌 영역의 큰 글씨(본문 1.2배 이상) 줄은 제목(##)으로 살린다.
    - <u> <sup> <mark> ~~ <br> 같은 서식 표시는 지운다(위 첨자 주석 번호는 글자로 남김).
[2] page.get_pixmap(dpi=200)으로 쪽 전체 PNG
[3] Gemini에 PNG + 'page_visual' 지시문 → 본문·일반 표가 아닌 차트·인포그래픽·흐름도만,
    항목마다 [graph] / [image] 표시, 제목, 위치(box=[ymin,xmin,ymax,xmax], 0~1000)를 붙여 답하게 한다. 없으면 '차트 없음'.
[4] '차트 없음'이 아니면 [1] 뒤에 graph·image 블록으로 붙인다.
    위치로 그림만 잘라 p{쪽}-g{n}.png / p{쪽}-i{n}.png로 저장하고, 쪽 전체 이미지(p{쪽}.png)도 남긴다.
    위치가 없거나 이상하면 쪽 전체 이미지를 쓰고 crop=page를 남긴다.
    같은 영역을 벡터로 읽은 색칠 칸 표가 있으면 비전 블록은 빼고 logs/vector_vs_vision.jsonl에 둘을 남긴다.

출력 표시(블록마다 종류를 남겨 청킹 때 메타데이터로 씀)
  <!-- page: N -->
  <!-- block: text|table|graph|image id=… page=N source=pymupdf4llm|rule|vector|vision
             [image="…png"] [page_image="…png"] [crop=page] --> … <!-- /block -->

사용법:
  python extract/extract_p4l.py                       # 01_data/additional/*.pdf 전부
  python extract/extract_p4l.py 01_data/additional/self_employed_debt_2025.pdf --pages 1-2
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf
import pymupdf4llm
from pymupdf4llm.ocr import OCRMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pdfplumber import Joiner  # noqa: E402  (줄 잇기 규칙 재사용)
from tables import extract_table  # noqa: E402
from spacing import SpacingFixer  # noqa: E402
import vision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DPI = 200
NO_VISUAL = "차트 없음"
VISION_BATCH = 10          # 렌더링한 쪽 이미지를 메모리에 이만큼씩만 들고 비전 호출

FORMAT_TAG_RE = re.compile(r"</?(u|mark|ins|del|s|span|font)(\s[^>]*)?>", re.I)
SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.I | re.S)
STRIKE_RE = re.compile(r"~~(.*?)~~", re.S)
BR_RE = re.compile(r"(\s*<br\s*/?>\s*)+", re.I)
VISUAL_HEAD_RE = re.compile(r"^\[(graph|image)\]\s*(.*)$", re.M)
TABLE_LINE_RE = re.compile(r"^\s*\|")


# ---- PyMuPDF → 표 로더가 쓰는 pdfplumber 모양으로 ------------------------------

class _PageShim:
    """PyMuPDF 글자로 pdfplumber식 단어를 만든다.

    PyMuPDF get_text('words')는 공백 글자로만 단어를 나눠서, 공백 없이 옆 칸에 놓인 글자('금액'|'비중*')가
    한 단어로 붙는다. pdfplumber처럼 글자 사이 간격(3pt 초과)으로도 나눈다.
    """
    GAP = 3.0

    SUP_RATIO = 0.8   # 줄에서 가장 큰 글자의 이 비율보다 작으면 위 첨자(주석 번호)로 본다

    def __init__(self, page: pymupdf.Page):
        self._words = []
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if any(not c["c"].isspace() for c in s["chars"])]
                main = max(spans, key=lambda s: (s["size"], len(s["chars"])), default=None)
                word: list = []
                for span in line["spans"]:
                    is_sup = main is not None and self._is_sup(span, main)
                    for ch in span["chars"]:
                        gap = word and ch["bbox"][0] - word[-1]["bbox"][2] > self.GAP
                        if ch["c"].isspace() or gap:
                            self._flush(word)
                            word = []
                        if not ch["c"].isspace():
                            word.append(dict(ch, sup=is_sup))
                self._flush(word)

    SUP_TEXT_RE = re.compile(r"^[\d\)\(\*,~]{1,5}$")

    def _is_sup(self, span: dict, main: dict) -> bool:
        """위 첨자 주석 번호: 글자가 작고, 기준선이 위로 올라가 있고, 숫자가 든 짧은 숫자·괄호·별표('2)', '*3').
        별표만 있는 것('비중*')은 첨자로 보지 않는다."""
        text = "".join(c["c"] for c in span["chars"]).strip()
        return (span["size"] < main["size"] * self.SUP_RATIO
                and span["origin"][1] < main["origin"][1] - main["size"] * 0.15
                and bool(self.SUP_TEXT_RE.match(text)) and any(ch.isdigit() for ch in text))

    def _flush(self, chars: list) -> None:
        if chars:
            # 위 첨자 주석 번호 앞에는 '^'를 넣는다: '(16.5' + 위첨자 '2)' → '(16.5^2)' ('(16.52)'로 읽히지 않게)
            text = "".join(("^" if c["sup"] and (i == 0 or not chars[i - 1]["sup"]) else "") + c["c"]
                           for i, c in enumerate(chars))
            self._words.append({"x0": min(c["bbox"][0] for c in chars), "top": min(c["bbox"][1] for c in chars),
                                "x1": max(c["bbox"][2] for c in chars), "bottom": max(c["bbox"][3] for c in chars),
                                "text": text,
                                # 표 로더가 칸 경계선에서 단어를 끊을 때 씀
                                "chars": [(c["bbox"][0], c["bbox"][2], c["c"]) for c in chars]})

    def extract_words(self):
        return self._words


class _TableShim:
    def __init__(self, tab):
        self.bbox = tuple(tab.bbox)
        self.cells = [tuple(c) for c in tab.cells if c]


def _overlap(a, b) -> float:
    """a 넓이 대비 a∩b 넓이."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def clean_format(text: str) -> str:
    # 위 첨자(주석 번호)는 '^'를 붙여 글자로 남긴다: '1.88%<sup>33)</sup>' → '1.88%^33)' (1.88%33)으로 붙지 않게)
    # '1)2)'처럼 번호가 여럿이면 번호마다: '^1)^2)'. 숫자가 없는 첨자(별표 등)는 첨자 표시 없이 글자만 둔다.
    def _sup(m):
        inner = re.sub(r"[*_\s]", "", m.group(1))
        if not inner:
            return ""
        if not re.search(r"\d", inner):
            return inner
        return re.sub(r"(\d{1,3}\))", r"^\1", inner) if re.fullmatch(r"(\d{1,3}\)){1,}", inner) else "^" + inner
    text = SUP_RE.sub(_sup, text)
    text = FORMAT_TAG_RE.sub("", text)
    text = STRIKE_RE.sub(r"\1", text)
    text = BR_RE.sub(" ", text)  # 칸 안 줄바꿈(<br>)은 공백으로(규칙 표와 같은 모양)
    text = re.sub(r"\$\^\{([^}$]*)\}\$", lambda m: "^" + m.group(1), text)   # 비전이 위 첨자를 수식으로 쓴 것: $^{1)}$ → ^1)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


PARA_START_RE = re.compile(r"^\s*(□|■|◦|○|●|⦿|▘|▪|•|-|\*|※|[①-⑳]|\d+\)|\d+\.|[가-하]\.|Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|[-])")
TEXT_BOX_MIN_HANGUL = 40      # 그림 상자 안 한글이 이만큼 이상이고
TEXT_BOX_MIN_RATIO = 0.35     # 공백 뺀 글자 중 한글 비율이 이 이상이면 '글이 든 그림 상자'
TITLE_SIZE_RATIO = 1.2        # 본문 글자 크기의 이 배수 이상이면 제목 후보
TITLE_WIDTH = 0.3             # 칸 구조 영역에서 제목으로 살릴 줄은 영역 너비의 이 비율 이상
SIDE_GAP = 8.0                # 같은 높이의 두 줄 사이 가로 간격이 이보다 크면 '나란히 놓인' 줄
CARD_MIN_LINES = 6            # 글자 상자가 카드(좌우 나란히)로 보이려면 필요한 줄 수
CARD_SIDE_RATIO = 0.6         # …그리고 나란히 놓인 줄의 비율
SIDE_RATIO = 0.2              # 줄 중 이 비율 이상이 나란히 놓였으면 칸 구조(도식·목록)로 본다
# 영역·비전 담김을 셀 때는 글자(한글·영문)가 든 단어만 센다. 차트 눈금 숫자('17 19 21')는 비전이 옮기지 않으므로 뺀다.
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]*[가-힣A-Za-z][가-힣A-Za-z0-9]*")


def region_lines(page: pymupdf.Page, bbox) -> list[tuple]:
    """그림 상자 안 줄 [(x0, y0, x1, y1, 글자, 글자크기)] (위→아래, 왼쪽→오른쪽)."""
    out = []
    for b in page.get_text("dict", clip=pymupdf.Rect(bbox))["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if spans:
                out.append((*l["bbox"], "".join(s["text"] for s in l["spans"]).strip(), max(s["size"] for s in spans)))
    return sorted(out, key=lambda r: (round(r[1]), r[0]))


def side_by_side(lines: list[tuple]) -> float:
    """다른 줄과 같은 높이에 가로로 떨어져 놓인 줄의 비율. 문장 단이면 0에 가깝고, 도식·목록이면 높다."""
    if len(lines) < 3:
        return 0.0
    hit = 0
    for i, a in enumerate(lines):
        for j, b in enumerate(lines):
            if i == j:
                continue
            overlap = min(a[3], b[3]) - max(a[1], b[1])
            if overlap >= 0.5 * min(a[3] - a[1], b[3] - b[1]) and max(a[0], b[0]) - min(a[2], b[2]) > SIDE_GAP:
                hit += 1
                break
    return hit / len(lines)


def prose_paragraphs(lines: list[tuple], joiner: Joiner, body_size: float) -> str:
    paras: list[str] = []
    title = False
    for _, _, _, _, line, size in lines:
        if body_size and size >= body_size * TITLE_SIZE_RATIO:   # 큰 글씨 줄은 제목으로 떼어 둔다
            paras.append(f"## {line}")
            title = True
        elif paras and not title and not PARA_START_RE.match(line):
            paras[-1] = joiner.join(paras[-1], line)
        else:
            paras.append(line)
            title = False
    return "\n\n".join(paras)


def region_titles(lines: list[tuple], bbox, body_size: float, top_only: bool) -> list[str]:
    hangul = lambda t: len(re.findall(r"[가-힣]", t))
    big = [l for l in lines if body_size and l[5] >= body_size * TITLE_SIZE_RATIO and hangul(l[4]) >= 2]
    if not top_only:                      # 글이 거의 없는 그림(배너 등): 큰 글씨 줄을 모두
        out: list[str] = []
        for l in big:
            # 차트 범례('적음')처럼 짧은 글자는 제목이 아니다
            if l[4] not in out and len(re.sub(r"\s", "", l[4])) >= 4:
                out.append(l[4])
        return out
    cand = [l for l in lines if hangul(l[4]) >= 2]
    if (cand and cand[0] in big and (cand[0][2] - cand[0][0]) >= TITLE_WIDTH * (bbox[2] - bbox[0])
            and len(re.sub(r"\s", "", cand[0][4])) >= 4):      # 차트 범례('적음')처럼 짧은 글자는 제목이 아니다
        t = cand[0]
        badge = next((l[4] for l in lines if re.fullmatch(r"\d{1,2}", l[4])
                      and min(l[3], t[3]) - max(l[1], t[1]) >= 0.5 * (t[3] - t[1]) and l[2] <= t[0] + 5), "")
        return [f"{badge} {t[4]}" if badge else t[4]]   # 칸 구조 영역: 맨 위의 넓은 큰 글씨 한 줄만(영역 제목)
    return []


def classify_picture(page: pymupdf.Page, bbox, joiner: Joiner, body_size: float):
    """그림으로 잡힌 상자를 가른다.

    prose : 글이 든 상자이고 줄이 한 줄기(나란히 놓인 줄이 거의 없음) → 글자층 글을 그대로 쓴다(설명 박스)
    multi : 글이 든 상자인데 줄이 나란히 놓임(도식·목록·흐름표) → 글자층 글은 쓰지 않고 비전에 맡긴다
    minor : 글이 거의 없는 그림(차트·로고·배너) → 큰 글씨 제목만 살린다
    반환: (종류, prose는 글 / 나머지는 제목 목록, 상자 안 단어 집합)
    """
    lines = region_lines(page, bbox)
    chars = re.sub(r"\s", "", "".join(l[4] for l in lines))
    hangul = len(re.findall(r"[가-힣]", chars))
    texty = hangul >= TEXT_BOX_MIN_HANGUL and hangul / max(len(chars), 1) >= TEXT_BOX_MIN_RATIO
    words = set(TOKEN_RE.findall(" ".join(l[4] for l in lines)))
    if texty and side_by_side(lines) < SIDE_RATIO:
        return "prose", prose_paragraphs(lines, joiner, body_size), words
    return ("multi" if texty else "minor"), region_titles(lines, bbox, body_size, top_only=texty), words


def body_font_size(doc: pymupdf.Document) -> float:
    sizes: dict[float, int] = {}
    for p in doc:
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    sizes[round(s["size"], 1)] = sizes.get(round(s["size"], 1), 0) + len(s["text"].strip())
    return max(sizes, key=sizes.get) if sizes else 10.0


def color_fills(page: pymupdf.Page) -> list[tuple]:
    """색 있는 채움 사각형(흰색·회색 제외) — 색칠 칸 표 판정용."""
    out = []
    for d in page.get_drawings():
        f, r = d.get("fill"), d["rect"]
        if f and len(f) >= 3 and max(f) - min(f) >= 0.15 and r.width > 2 and r.height > 2:
            out.append((r.x0, r.y0, r.x1, r.y1))
    return out


# ---- 2단 읽기 순서 -------------------------------------------------------------

COL_SEARCH = (0.3, 0.7)       # 단 경계를 찾는 쪽 너비 구간
NARROW = 0.55                 # 쪽 너비의 이 비율보다 좁은 상자만 단 상자로 본다
MIN_GUTTER = 6                # 단 사이 빈 띠 최소 너비(pt)


def reorder_columns(boxes: list[dict], width: float) -> list[dict]:
    """2단 쪽이면 상자를 '왼쪽 단 → 오른쪽 단' 순서로 다시 놓는다.

    pymupdf4llm은 상자를 위→아래(y)로 정렬해 두 단 상자가 번갈아 나온다.
    단 경계를 걸치는 넓은 상자(각주, 쪽 전체 제목·표)는 구분선으로 두고, 그 사이 구간마다 왼쪽 단 → 오른쪽 단.
    단 경계를 못 찾으면(1단 쪽) 순서를 바꾸지 않는다.
    """
    head = [b for b in boxes if b["class"] == "page-header"]
    foot = [b for b in boxes if b["class"] == "page-footer"]
    body = [b for b in boxes if b["class"] not in ("page-header", "page-footer")]
    if len(body) < 4:
        return boxes
    lo, hi = int(width * COL_SEARCH[0]), int(width * COL_SEARCH[1])
    # x마다 그 세로선을 걸치는 상자 수. 각주·넓은 제목처럼 걸치는 상자가 조금 있어도 되도록
    # '0개'가 아니라 '가장 적은' 곳을 찾고, 그 가장 긴 구간의 가운데를 단 경계로 삼는다.
    # 각주는 쪽 아래 넓게 놓이는 구분선이라 단 경계를 셀 때 뺀다(한국은행 양식은 각주가 여러 개)
    cols = [b for b in body if b["class"] != "footnote"]
    cross = {x: sum(1 for b in cols if b["bbox"][0] < x < b["bbox"][2]) for x in range(lo, hi)}
    least = min(cross.values())
    if least > max(2, 0.3 * len(cols)):
        return boxes
    best, run = None, []
    for x in list(range(lo, hi)) + [None]:
        if x is not None and cross[x] == least and run and x == run[-1] + 1:
            run.append(x)
            continue
        if run and len(run) >= MIN_GUTTER and (best is None or len(run) > len(best)):
            best = run
        run = [x] if x is not None and cross[x] == least else []
    if not best:
        return boxes
    gutter = (best[0] + best[-1]) / 2
    left = [b for b in body if b["bbox"][2] <= gutter]
    right = [b for b in body if b["bbox"][0] >= gutter]
    # 양쪽에 글 상자가 실제로 있어야 2단으로 본다(1단 쪽의 좌우 나란한 그림 두 개 등은 제외)
    texty = lambda bs: sum(1 for b in bs if b["class"] in ("text", "list-item", "section-header"))
    if len(left) < 2 or len(right) < 2 or texty(left) < 1 or texty(right) < 1:
        return boxes
    wide = sorted((b for b in body if b not in left and b not in right), key=lambda b: b["bbox"][1])
    seg = lambda b: sum(1 for w in wide if w["bbox"][1] <= b["bbox"][1])
    out = list(head)
    for k in range(len(wide) + 1):
        out += sorted((b for b in left if seg(b) == k), key=lambda b: b["bbox"][1])
        out += sorted((b for b in right if seg(b) == k), key=lambda b: b["bbox"][1])
        if k < len(wide):
            out.append(wide[k])
    return out + foot


def _png(page: pymupdf.Page, clip=None) -> bytes:
    return page.get_pixmap(dpi=DPI, clip=clip).tobytes("png")


# ---- 표 앞뒤의 제목·단위·주석 ---------------------------------------------------

CAPTION_RE = re.compile(r"^(표|<표|\[표|참고)\s*[\dⅠ-Ⅻ]")
BOLD_ONLY_RE = re.compile(r"^[#\s]*\*\*[^*]+\*\*\s*$")
UNIT_RE = re.compile(r"^\(?\s*단위")
NOTE_RE = re.compile(r"^(주\s*[:：)]|주\d|자료\s*[:：]|\*|※)")
FOOT_MARK_RE = re.compile(r"\^(\d{1,3})\)")          # 본문 위 첨자 '^33)' → '[^33]'
FOOT_REF_RE = re.compile(r"\[\^(\d{1,3})\]")
NUM_BANNER_RE = re.compile(r"^\*\*\d{1,2}\*\*\s+\S.{0,58}$")      # '**3** QR코드를 통한 편리한 정보 이용'


def _line_size(page: pymupdf.Page, bbox) -> float:
    """상자 안 글자 중 가장 큰 크기."""
    sizes = [sp["size"] for b in page.get_text("dict", clip=pymupdf.Rect(bbox))["blocks"]
             for ln in b.get("lines", []) for sp in ln["spans"] if sp["text"].strip()]
    return max(sizes, default=0)
FOOT_SPLIT_RE = re.compile(r"(?m)^>?\s*(\d{1,3})\)\s*")  # 각주 상자 안 '> 33) 내용'


def _strip_md(s: str) -> str:
    return re.sub(r"^[#>\-\s]+", "", s).replace("**", "").strip()


def _same_col(a: dict, b: dict) -> bool:
    ov = min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0])
    return ov >= 0.5 * min(a["bbox"][2] - a["bbox"][0], b["bbox"][2] - b["bbox"][0])


def table_context(boxes: list[dict], text: str) -> tuple[dict, set]:
    """표 바로 위의 제목·단위 줄과 바로 아래의 주석 줄(주:, 자료:, *, ※)을 표 블록에 붙이기 위해 찾는다.

    제목은 '표 Ⅲ-1.'처럼 번호가 있거나 줄 전체가 굵은 한 줄일 때만 쓴다(일반 소제목은 본문에 둔다).
    반환: ({id(표 상자): {caption, unit, notes}}, 표 블록으로 옮겨져 본문에서 빼야 할 상자 id 집합)
    """
    ctx, skip = {}, set()
    raw = lambda b: clean_format(text[b["pos"][0]:b["pos"][1]])
    for i, b in enumerate(boxes):
        if b["class"] != "table":
            continue
        c = {"caption": "", "unit": "", "notes": []}
        top, j = b["bbox"][1], i - 1
        while j >= 0 and id(boxes[j]) not in skip and -3 <= top - boxes[j]["bbox"][3] <= 40 and _same_col(boxes[j], b):
            p = boxes[j]
            r = raw(p)
            s = _strip_md(r)
            if p["class"] in ("text", "section-header", "caption", "list-item") and UNIT_RE.match(s) and not c["unit"]:
                c["unit"] = s
            elif (p["class"] in ("section-header", "caption", "text") and "\n" not in s and len(s) <= 60
                  and (CAPTION_RE.match(s) or BOLD_ONLY_RE.match(r.strip())) and not c["caption"]):
                c["caption"] = s
            else:
                break
            skip.add(id(p))
            top, j = p["bbox"][1], j - 1
        bottom, j = b["bbox"][3], i + 1
        while (j < len(boxes) and boxes[j]["class"] in ("caption", "text", "list-item")
               and -3 <= boxes[j]["bbox"][1] - bottom <= 30 and _same_col(boxes[j], b)):
            r = raw(boxes[j])
            if not NOTE_RE.match(_strip_md(r)):
                break
            c["notes"].append(re.sub(r"^[>\s]+", "", r))
            skip.add(id(boxes[j]))
            bottom, j = boxes[j]["bbox"][3], j + 1
        ctx[id(b)] = c
    return ctx, skip


def printed_numbers(texts: list[str]) -> list[str]:
    """머리말·꼬리말 글에서 인쇄 쪽 번호 후보(1~3자리 숫자, 글 맨 앞 또는 맨 뒤)."""
    out: list[str] = []
    for t in texts:
        s = re.sub(r"[>*#_\-–]", " ", clean_format(t)).strip()
        for m in (re.match(r"^(\d{1,3})(?:\s|$)", s), re.search(r"(?:^|\s)(\d{1,3})\s*$", s)):
            if m and m.group(1) not in out:
                out.append(m.group(1))
    return out


def column_sequence(boxes: list[dict], width: float) -> str:
    """점검용: 본문 상자를 순서대로 L(왼쪽 단) R(오른쪽 단) W(두 단에 걸침)로 적는다."""
    mid = width / 2
    return "".join("L" if b["bbox"][2] <= mid + 10 else "R" if b["bbox"][0] >= mid - 10 else "W"
                   for b in boxes if b["class"] not in ("page-header", "page-footer"))


# ---- 쪽 처리 ----------------------------------------------------------------

def build_page(page: pymupdf.Page, chunk: dict, joiner: Joiner, img_dir: Path, img_rel: str,
               body_size: float) -> dict:
    """[1] 결과를 블록 목록으로.

    반환: {blocks, retry(비전으로 다시 읽을 표 [(블록, png)]), regions(그림으로 잡힌 상자 기록),
           printed(인쇄 쪽 번호 후보), columns(점검용 단 순서)}
    """
    n = page.number + 1
    text = chunk["text"]
    found = page.find_tables().tables
    fills = color_fills(page)
    shim = _PageShim(page)
    blocks: list[dict] = []
    retry: list[tuple[dict, bytes]] = []
    regions: list[dict] = []
    hf: list[str] = []
    buf: list[str] = []
    first = {"k": None}                # 본문 버퍼가 시작된 상자 번호(읽는 순서)
    pics: list[dict] = []              # 그림 상자 [{index, bbox}] — 비전 블록이 들어갈 자리를 찾는 데 쓴다
    boxinfo: list[dict] = []           # 본문 상자 [{index, bbox}]

    def add(segment, k: int):
        if not buf:
            first["k"] = k
        buf.append(segment)
        for ln in str(segment).splitlines():
            if ln.startswith("#"):
                last["v"] = len(ln) - len(ln.lstrip("#"))

    def flush_text():
        body = clean_format("\n\n".join(s for s in buf if s.strip()))
        body = FOOT_MARK_RE.sub(r"[^\1]", body)     # 본문 첨자는 각주 표시 [^n]으로
        k0 = first["k"]
        buf.clear()
        if body:
            blocks.append({"type": "text", "source": "pymupdf4llm", "body": body, "order": k0})

    def next_tid() -> str:
        return f"p{n}-t{sum(b['type'] == 'table' for b in blocks) + 1}"

    # 같은 상자가 여러 번 나오는 경우(쪽 전체 그림이 3번 등) 한 번만, 2단 쪽은 단별 순서로
    seen, boxes = set(), []
    for b in chunk["page_boxes"]:
        key = (b["class"], tuple(b["bbox"]))
        if key not in seen:
            seen.add(key)
            boxes.append(b)
    boxes = reorder_columns(boxes, page.rect.width)
    columns = column_sequence(boxes, page.rect.width)
    ctx, skip = table_context(boxes, text)
    # 번호가 붙은 장 제목('# **1** …')의 단계. 번호 배지가 붙은 그림 영역 제목·번호 배너도 같은 단계로 맞춘다.
    num_levels = [len(m.group(1)) for b in boxes if b["class"] == "section-header"
                  for m in [re.match(r"^(#+)\s*\*\*\d{1,2}\*\*", text[b["pos"][0]:b["pos"][1]].strip())] if m]
    num_level = Counter(num_levels).most_common(1)[0][0] if num_levels else 2
    last = {"v": 0}                    # 지금까지 나온 마지막 제목의 단계

    for k, box in enumerate(boxes):
        if id(box) in skip:
            continue
        cls = box["class"]
        seg = text[box["pos"][0]:box["pos"][1]]
        if cls in ("page-header", "page-footer") and n != 1:
            # 머리말·꼬리말은 본문에서 빼고 쪽 번호만 읽는다. 첫 쪽(표지)은 문서 제목·날짜·발행기관이
            # 머리말로 분류되는 경우가 있어 본문으로 남긴다.
            hf.append(seg)
            continue
        if cls not in ("footnote", "page-header", "page-footer"):
            boxinfo.append({"index": k, "bbox": tuple(box["bbox"])})
        if cls == "footnote":
            body = clean_format(seg)
            marks = list(FOOT_SPLIT_RE.finditer(body))
            if not marks:
                add(seg, k)
                continue
            flush_text()
            for k, m in enumerate(marks):
                end = marks[k + 1].start() if k + 1 < len(marks) else len(body)
                blocks.append({"type": "footnote", "id": f"p{n}-fn{m.group(1)}", "number": m.group(1),
                               "source": "pymupdf4llm", "body": f"[^{m.group(1)}]: {body[m.end():end].strip()}",
                               "order": k})
            continue
        if cls == "picture":
            flush_text()             # 그림 앞의 본문은 끊는다(비전 블록이 그림 자리에 들어가도록)
            pics.append({"index": k, "bbox": tuple(box["bbox"])})
            # 그림으로 잡혔어도 안에 색칠 칸 표(간트형)가 있으면 좌표로 읽는다
            done = False
            for tab in [t for t in found if _overlap(t.bbox, box["bbox"]) >= 0.5]:
                res = extract_table(shim, _TableShim(tab), joiner.join, fills)
                if res.kind == "range":
                    flush_text()
                    blocks.append({"type": "table", "id": next_tid(), "source": "vector",
                                   "body": res.to_md(), "bbox": tuple(tab.bbox), "order": k})
                    found = [t for t in found if t is not tab]
                    done = True
            if done:
                continue
            kind, payload, words = classify_picture(page, box["bbox"], joiner, body_size)
            if kind == "prose":      # 설명 박스: 글자층 글이 원문 그대로라 이쪽이 정확하다
                add(payload, k)
                regions.append({"id": f"p{n}-r{len(regions) + 1}", "bbox": tuple(box["bbox"]), "kind": "prose",
                                "words": words, "index": k})
            else:                    # 도식·목록(multi)은 글자 순서가 섞이므로 제목만 살리고 비전에 맡긴다
                for t in payload:
                    badge = re.match(r"^(\d{1,2})\s+(.*)$", t)
                    if kind == "multi" and badge:        # 번호 배지가 붙은 제목: 같은 쪽 번호 제목과 같은 단계
                        add("#" * num_level + f" **{badge.group(1)}** {badge.group(2)}", k)
                    elif kind == "multi":                # 소제목: 바로 앞 제목보다 한 단계 아래
                        add("#" * min(6, (last["v"] or 1) + 1) + f" {t}", k)
                    else:
                        add(f"## {t}", k)
                if kind == "multi":
                    regions.append({"id": f"p{n}-r{len(regions) + 1}", "bbox": tuple(box["bbox"]),
                                    "kind": "multi", "words": words, "index": k})
            flush_text()             # 그림 제목은 그림 자리 바로 앞에서 끊는다
            continue
        if cls in ("text", "list-item"):
            # 글자 상자 안에서 줄이 좌우로 나란히 놓인 카드·비교 박스(단기/중기/장기 등)는 글자 순서가 섞이므로
            # 도식 영역(multi)으로 보고 비전에 맡긴다
            ls = region_lines(page, box["bbox"])
            if len(ls) >= CARD_MIN_LINES and side_by_side(ls) >= CARD_SIDE_RATIO:
                flush_text()
                pics.append({"index": k, "bbox": tuple(box["bbox"])})
                words = set(TOKEN_RE.findall(" ".join(l[4] for l in ls)))
                regions.append({"id": f"p{n}-r{len(regions) + 1}", "bbox": tuple(box["bbox"]),
                                "kind": "multi", "words": words, "index": k})
                continue
        if cls != "table":
            one = clean_format(seg).strip()
            if (cls == "text" and NUM_BANNER_RE.match(one) and "\n" not in one
                    and _line_size(page, box["bbox"]) >= body_size * TITLE_SIZE_RATIO):
                add("#" * num_level + " " + one, k)     # 번호 배너 → 같은 쪽 번호 제목과 같은 단계
            else:
                add(seg, k)
            continue
        flush_text()
        made: list[dict] = []
        # 레이아웃 표 상자 안에 든 PyMuPDF 표(여러 조각으로 찾을 때도 있음)를 위→아래로 모두 쓴다
        parts = sorted((t for t in found if _overlap(t.bbox, box["bbox"]) >= 0.5), key=lambda t: t.bbox[1])
        # PyMuPDF 표가 레이아웃 표 상자의 일부만 잡았으면(괘선 없는 머리글 행 등) 상자 전체를 비전으로 읽는다
        inside = lambda w, bb: bb[0] <= (w["x0"] + w["x1"]) / 2 <= bb[2] and bb[1] <= (w["top"] + w["bottom"]) / 2 <= bb[3]
        leftover = [w for w in shim.extract_words() if inside(w, box["bbox"])
                    and not any(inside(w, tuple(t.bbox)) for t in parts)]
        if not parts:
            made.append({"type": "table", "id": next_tid(), "source": "pymupdf4llm", "body": clean_format(seg),
                         "bbox": tuple(box["bbox"])})
        elif len(leftover) > 2:
            tid = next_tid()
            png = _png(page, clip=pymupdf.Rect(box["bbox"]) + (-4, -4, 4, 4))
            block = {"type": "table", "id": tid, "source": "vision", "image": f"{img_rel}/{tid}.png",
                     "body": clean_format(seg), "problems": [f"표 일부만 검출(남은 단어 {len(leftover)}개)"],
                     "bbox": tuple(box["bbox"])}
            (img_dir / f"{tid}.png").write_bytes(png)
            retry.append((block, png))
            made.append(block)
            found = [t for t in found if t not in parts]
        else:
            for tab in parts:
                tid = next_tid() if not made else f"p{n}-t{sum(b['type'] == 'table' for b in blocks) + len(made) + 1}"
                res = extract_table(shim, _TableShim(tab), joiner.join, fills)
                block = {"type": "table", "id": tid, "source": "vector" if res.kind == "range" else "rule",
                         "body": res.to_md(), "problems": res.problems, "bbox": tuple(tab.bbox)}
                if res.problems or not res.rows:
                    png = _png(page, clip=pymupdf.Rect(tab.bbox) + (-4, -4, 4, 4))
                    block.update(source="vision", image=f"{img_rel}/{tid}.png")
                    (img_dir / f"{tid}.png").write_bytes(png)
                    retry.append((block, png))
                made.append(block)
            found = [t for t in found if t not in parts]  # 다음 표 상자가 같은 조각을 다시 쓰지 않게
        c = ctx.get(id(box), {})
        if c and made:
            made[0]["caption"], made[0]["unit"] = c["caption"], c["unit"]
            made[-1]["notes"] = c["notes"]
        for b in made:
            b["order"] = k
        blocks.extend(made)
    flush_text()
    # 각주 블록이 없는 번호(표·그림 아래 '주: 1) …'의 번호)는 각주 표시 [^n]이 아니라 원래 '^n)'로 되돌린다
    foot_nums = {b["number"] for b in blocks if b["type"] == "footnote"}
    for b in blocks:
        if b["type"] == "text":
            b["body"] = FOOT_REF_RE.sub(lambda m: m.group(0) if m.group(1) in foot_nums else f"^{m.group(1)})", b["body"])
    return {"blocks": blocks, "retry": retry, "regions": regions, "printed": printed_numbers(hf), "columns": columns,
            "pics": pics, "boxinfo": boxinfo}


BOX_RE = re.compile(r"\s*\|?\s*box\s*=\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*$", re.I)
CROP_PAD = 0.015              # 잘라낼 때 쪽 크기 대비 여백
MIN_CROP_AREA = 0.01          # 이보다 작은 box는 믿지 않고 쪽 전체 이미지를 쓴다


def parse_visuals(answer: str) -> list[dict]:
    """'[graph] 제목 | box=[ymin, xmin, ymax, xmax]\\n…' 형식 응답 → [{kind, title, body, box}]"""
    answer = answer.strip()
    if not answer or answer.startswith(NO_VISUAL):
        return []
    heads = list(VISUAL_HEAD_RE.finditer(answer))
    if not heads:
        return [{"kind": "image", "title": "", "body": answer, "box": None}]
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(answer)
        title, box = m.group(2).strip(), None
        if bm := BOX_RE.search(title):
            box = tuple(float(v) for v in bm.groups())
            title = title[:bm.start()].strip()
        rest = answer[m.end():end].strip().strip("-").strip()
        out.append({"kind": m.group(1), "title": title, "box": box,
                    "body": (f"**{title}**\n\n" if title else "") + rest})
    return out


def box_to_rect(box, page: pymupdf.Page) -> pymupdf.Rect | None:
    """0~1000 정규화 [ymin, xmin, ymax, xmax] → 쪽 좌표 사각형(여백 포함). 이상하면 None."""
    if not box:
        return None
    ymin, xmin, ymax, xmax = box
    if not (0 <= xmin < xmax <= 1000 and 0 <= ymin < ymax <= 1000):
        return None
    if (xmax - xmin) * (ymax - ymin) / 1e6 < MIN_CROP_AREA:
        return None
    w, h = page.rect.width, page.rect.height
    r = pymupdf.Rect(xmin / 1000 * w - CROP_PAD * w, ymin / 1000 * h - CROP_PAD * h,
                     xmax / 1000 * w + CROP_PAD * w, ymax / 1000 * h + CROP_PAD * h)
    return r & page.rect


def block_md(b: dict, page_no: int) -> str:
    attrs = f"{b['type']}" + (f" id={b['id']}" if b.get("id") else "") + f" page={page_no} source={b['source']}"
    if b.get("number"):
        attrs += f" number={b['number']}"
    if b.get("image"):
        attrs += f' image="{b["image"]}"'
    if b.get("page_image"):
        attrs += f' page_image="{b["page_image"]}"'
    if b.get("crop"):
        attrs += f" crop={b['crop']}"
    if b.get("continues"):
        attrs += f" continues={b['continues']}"
    if b.get("continued_from"):
        attrs += f" continued_from={b['continued_from']}"
    if b.get("title"):
        attrs += ' title="' + b["title"].replace('"', "'").replace("--", "-") + '"'
    parts = [f"<!-- block: {attrs} -->"]
    if b.get("caption"):
        parts.append(f"**{b['caption']}**")
    if b.get("unit"):
        parts.append(b["unit"])
    parts.append(b["body"].strip())
    parts += b.get("notes") or []
    parts.append("<!-- /block -->")
    return "\n\n".join(parts)


def anchor_order(pd: dict, rect) -> float:
    """비전 블록이 들어갈 자리(레이아웃 상자 순서 번호).

    그림 상자와 많이(30% 이상) 겹치면 그 그림 상자 바로 뒤, 아니면 같은 단에서 바로 위에 있는 상자 뒤,
    위치를 모르면(쪽 전체 이미지로 대신한 경우) 쪽의 맨 끝.
    """
    if rect is None:
        return 10 ** 6
    r = tuple(rect)
    best, frac = None, 0.0
    for pic in pd["pics"]:
        f = _overlap(r, pic["bbox"])
        if f > frac:
            best, frac = pic, f
    if best is not None and frac >= 0.3:
        return best["index"] + 0.1
    w = r[2] - r[0]
    above = [(bi["bbox"][3], bi["index"]) for bi in pd["boxinfo"]
             if bi["bbox"][3] <= r[1] + 5
             and min(r[2], bi["bbox"][2]) - max(r[0], bi["bbox"][0]) >= 0.5 * min(w, bi["bbox"][2] - bi["bbox"][0])]
    return max(above)[1] + 0.5 if above else -1


def resolve_printed(doc: pymupdf.Document, cands: dict[int, list[str]]) -> dict[int, list[str]]:
    """쪽마다 인쇄 쪽 번호 후보가 둘 이상이면(장 번호가 섞임) 문서 전체에서 가장 흔한 (PDF쪽 - 인쇄쪽) 차이에 맞는 것을 고른다.
    가로로 두 쪽을 붙인 펼침면은 후보를 모두 남긴다."""
    offs = Counter(n - int(c[0]) for n, c in cands.items() if len(c) == 1)
    mode = offs.most_common(1)[0][0] if offs else None
    out = {}
    for n, c in cands.items():
        p = doc[n - 1]
        if len(c) > 1 and mode is not None and p.rect.width <= p.rect.height * 1.2:
            c = [x for x in c if n - int(x) == mode] or c
        out[n] = sorted(c, key=int)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s", "", s)


def _head_key(s: str) -> str:
    """제목 비교용: 앞의 '#'과 장 번호('2 ')를 뗀 글자."""
    return _norm(re.sub(r"^\*{0,2}\d{1,2}\*{0,2}\s+", "", re.sub(r"^#+\s*", "", s.strip())))


def _first_title(body: str) -> str:
    """비전 결과 본문에서 제목으로 쓸 첫 줄(마크다운 기호를 뗀 것). 없으면 빈 글자."""
    if body.lstrip().startswith("|"):
        return ""
    for ln in body.splitlines():
        t = re.sub(r"^[#>\-*\s]+", "", ln).replace("**", "").strip()
        # 번호 글자('1')나 너무 짧은 줄은 제목이 아니다
        if t and not t.startswith("|") and len(t) >= 4 and not re.fullmatch(r"[\d\s.()]+", t):
            return t[:80]
    return ""


def drop_repeated_lines(blocks: list[dict]) -> list[dict]:
    """비전이 그림 주변 글(소제목, 주:/자료: 줄, 바로 아래 목록)까지 읽어 본문과 겹친 줄을 지운다.

    본문(text) 블록의 줄과, 앞선 비전 블록의 줄과 똑같은(마커·공백 무시, 8자 이상) 줄은 뒤쪽 비전 블록에서 뺀다.
    원문 그대로인 본문 쪽을 남긴다. 표·각주 블록은 건드리지 않는다.
    """
    key = lambda line: re.sub(r"[\s*#|\-:>\[\]^]", "", line)
    seen = {key(ln) for b in blocks if b["type"] == "text" for ln in b["body"].splitlines() if len(key(ln)) >= 8}
    out = []
    for b in blocks:
        if b["source"] == "vision" and b["type"] in ("graph", "image"):
            kept = []
            for ln in b["body"].splitlines():
                k = key(ln)
                if len(k) >= 8 and k in seen:
                    continue
                if len(k) >= 8:
                    seen.add(k)
                kept.append(ln)
            body = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
            if not TOKEN_RE.search(body):
                continue
            b = {**b, "body": body}
        out.append(b)
    return out


def _coverage(words: set, text_norm: str) -> float:
    return sum(1 for w in words if w in text_norm) / len(words) if words else 1.0


def run_checks(doc: pymupdf.Document, page_data: dict, page_blocks: dict) -> list[str]:
    """추출 뒤 자동 점검: 단 순서 섞임 / 글이 든 그림 영역 단어 빠뜨림 / 한 쪽 안 같은 문장 중복."""
    issues = []
    for n in sorted(page_blocks):
        blocks = page_blocks[n]
        lv = {len(m.group(1)) for b in blocks if b["type"] == "text" for ln in b["body"].splitlines()
              for m in [re.match(r"^(#+)\s*\*\*\d{1,2}\*\*", ln)] if m}
        if len(lv) > 1:
            issues.append(f"p{n} 번호 제목 단계가 서로 다름 {sorted(lv)}")
        seq = page_data[n]["columns"]
        if any("RL" in s for s in seq.split("W")):
            issues.append(f"p{n} 단 순서 섞임 ({seq})")
        out = _norm(" ".join(b["body"] for b in blocks))
        for r in page_data[n]["regions"]:
            cov = _coverage(r["words"], out)
            if cov < 0.9:
                issues.append(f"p{n} {r['id']}({r['kind']}) 영역 단어 {cov:.0%}만 담김")
        seen: dict[str, int] = {}
        for bi, b in enumerate(blocks):
            if b["type"] in ("table", "footnote"):
                continue
            for line in b["body"].splitlines():
                key = re.sub(r"[\s*#|\-:>\[\]^]", "", line)
                if len(key) >= 20:
                    if key in seen and seen[key] != bi:
                        issues.append(f"p{n} 같은 문장이 두 블록에 있음: {line.strip()[:36]}")
                    seen.setdefault(key, bi)
    return issues


P4L_CACHE = ROOT / "cache" / "p4l"
P4L_OPTIONS = "page_chunks=True use_ocr=NEVER force_text=False"


def get_chunks(doc: pymupdf.Document, pdf_path: Path, pages: list[int] | None) -> list[dict]:
    """pymupdf4llm 쪽별 결과. 레이아웃 분석이 느려(쪽당 1~2초) 쪽 단위로 캐시한다.

    캐시 키: PDF 내용 해시 + pymupdf4llm 버전 + 옵션. PDF나 버전이 바뀌면 자동으로 새로 만든다.
    pages: 0부터 시작하는 쪽 번호 목록(None이면 전체).
    """
    import hashlib
    import pickle
    key = hashlib.sha256(pdf_path.read_bytes() + pymupdf4llm.__version__.encode() + P4L_OPTIONS.encode()).hexdigest()[:16]
    cache_dir = P4L_CACHE / f"{pdf_path.stem}_{key}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(pages) if pages is not None else list(range(doc.page_count))
    missing = [p for p in wanted if not (cache_dir / f"{p + 1}.pkl").exists()]
    if missing:
        fresh = pymupdf4llm.to_markdown(doc, pages=missing, page_chunks=True, use_ocr=OCRMode.NEVER,
                                        force_text=False, show_progress=False)
        for c in fresh:
            n = c["metadata"]["page_number"]
            with (cache_dir / f"{n}.pkl").open("wb") as fp:
                pickle.dump(dict(c), fp)   # defaultdict(lambda)는 pickle이 안 되므로 dict로
    out = []
    for p in wanted:
        with (cache_dir / f"{p + 1}.pkl").open("rb") as fp:
            out.append(pickle.load(fp))
    print(f"  [1] pymupdf4llm {len(out)} pages (new {len(missing)}, cached {len(out) - len(missing)})", flush=True)
    return out


def _table_like(b: dict) -> bool:
    return b.get("bbox") is not None and (b["type"] == "table" or (b["type"] == "image" and b["body"].lstrip().startswith("|")))


def mark_continuations(doc: pymupdf.Document, page_blocks: dict) -> list[tuple[str, str]]:
    """표가 쪽 아래에서 끝나고 다음 쪽 위에서 이어지는 것으로 보이면 두 블록에 표시를 남긴다.

    조건: 앞 쪽의 가장 아래 표 블록의 아래쪽이 쪽 높이의 80% 이상, 다음 쪽의 가장 위 표 블록의 위쪽이 20% 이하.
    (표시만 하고 합치지는 않는다. 합치는 일은 청킹 단계에서 한다.)
    """
    marks = []
    for n in sorted(page_blocks):
        if n + 1 not in page_blocks:
            continue
        h = doc[n - 1].rect.height
        prev = [b for b in page_blocks[n] if _table_like(b)]
        nxt = [b for b in page_blocks[n + 1] if _table_like(b)]
        if not prev or not nxt:
            continue
        last, first = max(prev, key=lambda b: b["bbox"][3]), min(nxt, key=lambda b: b["bbox"][1])
        if last["bbox"][3] >= 0.8 * h and first["bbox"][1] <= 0.2 * h and not last.get("id", "").startswith("p0"):
            last["continues"], first["continued_from"] = first.get("id", ""), last.get("id", "")
            marks.append((last.get("id", ""), first.get("id", "")))
    return marks


def outline(page_blocks: dict) -> list[str]:
    """그림이 있는 쪽마다 블록 순서를 한 줄로 적는다. 그림이 자기 제목 뒤에 있는지 눈으로 확인하는 용도."""
    out = []
    for n in sorted(page_blocks):
        bl = page_blocks[n]
        if not any(b["source"] == "vision" and b["type"] in ("graph", "image") for b in bl):
            continue
        parts: list[str] = []
        for b in bl:
            if b["type"] == "text":
                body_run = False
                for ln in b["body"].splitlines():
                    if ln.startswith("#"):
                        parts.append(ln.strip()[:32])
                        body_run = False
                    elif ln.strip() and not body_run:
                        parts.append("(본문)")
                        body_run = True
            elif b["type"] == "footnote":
                continue
            elif b["type"] == "table":
                parts.append(f"[표 {b.get('id', '')}]")
            else:
                parts.append(f"[{b['type']} {b.get('id', '')} {(b.get('title') or '')[:18]}]")
        out.append(f"p{n}: " + " | ".join(parts))
    return out


def convert(pdf_path: Path, out_dir: Path, pages: list[int] | None = None, cache_only: bool = False) -> tuple[str, dict]:
    img_rel = f"images/{pdf_path.stem}"
    img_dir = out_dir / img_rel
    img_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    joiner = Joiner([l for p in doc for l in p.get_text().splitlines()])
    body_size = body_font_size(doc)

    chunks = get_chunks(doc, pdf_path, pages)
    if cache_only:
        return "", {}

    page_data: dict[int, dict] = {}
    for c in chunks:
        n = c["metadata"]["page_number"]
        page_data[n] = build_page(doc[n - 1], c, joiner, img_dir, img_rel, body_size)
    page_blocks = {n: d["blocks"] for n, d in page_data.items()}
    retry = [r for d in page_data.values() for r in d["retry"]]
    printed = resolve_printed(doc, {n: d["printed"] for n, d in page_data.items() if d["printed"]})

    # 표 다시 읽기(검사 실패)
    if retry:
        for (b, _), ans in zip(retry, vision.describe_many([(png, "table") for _, png in retry])):
            b["body"] = clean_format(ans)

    # [2][3][4] 쪽 전체 이미지 → 시각 자료
    t = time.time()
    nums = sorted(page_blocks)
    pending: list[tuple[int, dict]] = []     # 비전이 읽지 않은 도식·목록 영역(다시 잘라 읽는다)
    dropped = {"vector": 0, "prose": 0}
    for i in range(0, len(nums), VISION_BATCH):
        batch = nums[i:i + VISION_BATCH]
        pngs = [_png(doc[n - 1]) for n in batch]
        for n, png, ans in zip(batch, pngs, vision.describe_many([(p, "page_visual") for p in pngs])):
            visuals = parse_visuals(clean_format(ans))
            page = doc[n - 1]
            regs = page_data[n]["regions"]
            if visuals:
                page_img = f"{img_rel}/p{n}.png"
                (img_dir / f"p{n}.png").write_bytes(png)
                vector_tables = [b for b in page_blocks[n] if b.get("source") == "vector"]
                prose_words = [r["words"] for r in regs if r["kind"] == "prose"]
                heads = {_head_key(ln) for b in page_blocks[n] if b["type"] == "text"
                         for ln in b["body"].splitlines() if ln.startswith("#")}
                counts = {"graph": 0, "image": 0}
                for v in visuals:
                    rect = box_to_rect(v["box"], page)
                    # 같은 영역을 벡터 좌표로 이미 읽은 색칠 칸 표가 있으면 벡터 결과를 쓰고 비전 블록은 뺀다
                    dup = next((t for t in vector_tables if rect is not None
                                and _overlap(t["bbox"], tuple(rect)) >= 0.5), None)
                    if dup is None:
                        dup = next((t for t in vector_tables if v["title"] and v["title"][:8] in page.get_text()
                                    and _title_near(page, v["title"], t["bbox"])), None)
                    if dup is not None:
                        _log_vector_vs_vision(pdf_path.name, n, dup, v)
                        dropped["vector"] += 1
                        continue
                    # 글자층 설명 박스(prose)와 내용이 같으면 원문 그대로인 글자층을 남기고 비전은 뺀다
                    vw = set(TOKEN_RE.findall(v["body"]))
                    if vw and any(len(vw & pw) / len(vw) >= 0.8 for pw in prose_words):
                        dropped["prose"] += 1
                        continue
                    body = v["body"]
                    if v["title"] and _head_key(v["title"]) in heads:       # 같은 제목이 본문에 ## 로 이미 있으면 굵은 제목은 뺀다
                        body = body.split("\n\n", 1)[1] if "\n\n" in body else ""
                    counts[v["kind"]] += 1
                    vid = f"p{n}-{v['kind'][0]}{counts[v['kind']]}"
                    block = {"type": v["kind"], "id": vid, "source": "vision", "body": body, "page_image": page_img,
                             "rect": tuple(rect) if rect is not None else None,
                             "order": anchor_order(page_data[n], rect), "title": v["title"]}
                    if rect is not None:
                        (img_dir / f"{vid}.png").write_bytes(_png(page, clip=rect))
                        block["image"] = f"{img_rel}/{vid}.png"
                    else:
                        block.update(image=page_img, crop="page")
                    page_blocks[n].append(block)
            # 도식·목록 영역을 비전이 얼마나 담았나: 80% 미만이면 그 영역만 잘라 다시 읽는다
            seen_text = _norm(" ".join(b["body"] for b in page_blocks[n] if b.get("source") == "vision"))
            for r in regs:
                r["page"] = n
                if r["kind"] == "multi":
                    r["coverage"] = round(_coverage(r["words"], seen_text), 2)
                    r["handled"] = "page-vision" if r["coverage"] >= 0.8 else "re-read"
                    if r["coverage"] < 0.8:
                        pending.append((n, r))
                else:
                    r["handled"] = "text-layer"
        print(f"  [3] vision {min(i + VISION_BATCH, len(nums))}/{len(nums)} pages", flush=True)
    print(f"  [3] done ({time.time() - t:.0f}s) dropped vision blocks: {dropped}", flush=True)

    if pending:
        jobs = []
        for n, r in pending:
            png = _png(doc[n - 1], clip=pymupdf.Rect(r["bbox"]) + (-4, -4, 4, 4))
            (img_dir / f"{r['id']}.png").write_bytes(png)
            jobs.append((png, "figure"))
        print(f"  [3b] region re-read {len(jobs)}", flush=True)
        for (n, r), ans in zip(pending, vision.describe_many(jobs)):
            body = clean_format(ans)
            if body and not body.strip().startswith("내용 없음"):
                # 다시 읽은 영역이 기존 비전 블록을 통째로 품고 있으면(블록 넓이의 80% 이상이 영역 안) 그 블록은 새 결과로 대체한다.
                inside = [b for b in page_blocks[n] if b["source"] == "vision" and b["type"] in ("graph", "image")
                          and b.get("rect") and _overlap(b["rect"], r["bbox"]) >= 0.8]
                page_blocks[n] = [b for b in page_blocks[n] if b not in inside]
                r["replaced"] = [b["id"] for b in inside]
                page_blocks[n].append({"type": "image", "id": r["id"], "source": "vision", "body": body,
                                       "image": f"{img_rel}/{r['id']}.png", "crop": "region",
                                       "order": r["index"] + 0.1, "title": _first_title(body),
                                       "bbox": tuple(r["bbox"])})
            else:
                r["handled"] = "re-read-empty"

    for n in nums:
        # 읽는 순서(레이아웃 상자 순서)대로 놓는다. 같은 번호끼리는 만든 순서를 지킨다.
        page_blocks[n].sort(key=lambda b: b.get("order", 10 ** 6))
        page_blocks[n] = drop_repeated_lines(page_blocks[n])
        # 각주 블록은 쪽의 맨 끝으로
        page_blocks[n] = [b for b in page_blocks[n] if b["type"] != "footnote"] + \
                         [b for b in page_blocks[n] if b["type"] == "footnote"]
    marks = mark_continuations(doc, page_blocks)
    if marks:
        print(f"  continues: {marks}", flush=True)

    # 본문·각주 띄어쓰기 정리(표·비전 결과는 제외). 바뀐 내용은 logs/spacing_<문서>.json에 남겨 검토할 수 있게 한다.
    fixer = SpacingFixer()
    fixer.set_evidence([b["body"] for n in nums for b in page_blocks[n] if b["type"] in ("text", "footnote")])
    for n in nums:
        for b in page_blocks[n]:
            if b["type"] in ("text", "footnote"):
                b["body"] = fixer.fix(b["body"])
    import json
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / f"spacing_{pdf_path.stem}.json").write_text(
        json.dumps([list(c) for c in fixer.changes], ensure_ascii=False), encoding="utf-8")
    print(f"  spacing: {dict(Counter(k for k, _, _ in fixer.changes))}", flush=True)

    lines_outline = outline(page_blocks)
    (ROOT / "logs" / f"outline_{pdf_path.stem}.txt").write_text("\n".join(lines_outline), encoding="utf-8")
    # 그림으로 잡힌 상자(글이 든 것)마다 어떻게 처리됐는지 남긴다(검토용)
    regions_log = [{"page": r["page"], "id": r["id"], "kind": r["kind"], "handled": r["handled"],
                    "coverage": r.get("coverage"), "bbox": [round(v) for v in r["bbox"]],
                    "replaced": r.get("replaced", [])}
                   for d in page_data.values() for r in d["regions"] if "page" in r]
    (ROOT / "logs" / f"regions_{pdf_path.stem}.json").write_text(
        json.dumps(regions_log, ensure_ascii=False, indent=1), encoding="utf-8")
    issues = run_checks(doc, page_data, page_blocks)
    print(f"  checks: {len(issues)} issues", flush=True)
    for msg in issues[:60]:
        print(f"    - {msg}", flush=True)

    stats: dict[str, int] = {}
    out = [f"<!-- source: {pdf_path.name} -->",
           f"<!-- extracted: {time.strftime('%Y-%m-%d')} extractor=extract_p4l.py "
           f"pymupdf4llm={pymupdf4llm.__version__} vision={vision.MODEL} -->"]
    for n in nums:
        pr = printed.get(n)
        out.append(f"<!-- page: {n}" + (f" (printed {','.join(pr)})" if pr else "") + " -->")
        for b in page_blocks[n]:
            key = f"{b['type']}_{b['source']}"
            stats[key] = stats.get(key, 0) + 1
            out.append(block_md(b, n))
    return "\n\n".join(out) + "\n", stats


LOG = ROOT / "logs" / "vector_vs_vision.jsonl"
REG_DIR = ROOT / "logs" / "regression_md"
# 문제였던 쪽(1부터). 코드를 고칠 때 먼저 이 쪽만 돌려 확인한다.
REGRESSION = {
    "housing_pension_2026": [26, 41, 72, 81, 85],
    "insurance_terms_example_2020": [2, 4, 5, 6, 7, 8],
    "housing_policy_finance_2025": [3, 6, 12, 14, 17],
    "policy_finance_debt_adjustment_2026": [66],
    "retirement_pension_2025": [2, 4],
    "self_employed_debt_2025": [2],
}


def _title_near(page: pymupdf.Page, title: str, bbox) -> bool:
    """비전 제목이 벡터 표 바로 위(80pt 안)에 적혀 있는지."""
    for r in page.search_for(title[:12]):
        if 0 <= bbox[1] - r.y1 <= 80:
            return True
    return False


def _log_vector_vs_vision(pdf_name: str, page_no: int, table: dict, visual: dict) -> None:
    """같은 그림을 벡터·비전이 둘 다 읽은 경우 둘을 남긴다(비전 오류율 점검용)."""
    import json
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"file": pdf_name, "page": page_no, "table_id": table["id"],
                             "vector": table["body"], "vision": visual["body"]}, ensure_ascii=False) + "\n")


def _parse_pages(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a) - 1, int(b or a)))  # pymupdf4llm pages는 0부터
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, default=ROOT / "md" / "additional_p4l")
    ap.add_argument("--pages", help="예: 1-3,7 (시험용)")
    ap.add_argument("--cache-only", action="store_true", help="pymupdf4llm 캐시만 채우고 끝낸다(비전 호출 없음)")
    ap.add_argument("--regression", action="store_true", help="문제였던 쪽만 돌려 logs/regression_md/에 저장하고 점검 결과를 출력")
    args = ap.parse_args()
    if args.regression:
        REG_DIR.mkdir(parents=True, exist_ok=True)
        for name, pgs in REGRESSION.items():
            print(f"{name} pages {pgs}", flush=True)
            md, stats = convert(ROOT / "01_data" / "additional" / f"{name}.pdf", REG_DIR, [p - 1 for p in pgs])
            (REG_DIR / f"{name}.md").write_text(md, encoding="utf-8")
            print("  outline:", flush=True)
            for ln in (ROOT / "logs" / f"outline_{name}.txt").read_text(encoding="utf-8").splitlines():
                print("    " + ln, flush=True)
            print(f"  blocks {stats}", flush=True)
        return
    pdfs = args.pdfs or sorted((ROOT / "01_data" / "additional").glob("*.pdf"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        print(f"{pdf.name}", flush=True)
        md, stats = convert(pdf, args.out_dir, _parse_pages(args.pages), args.cache_only)
        if args.cache_only:
            continue
        out = args.out_dir / (pdf.stem + ".md")
        out.write_text(md, encoding="utf-8")
        print(f"  blocks {stats} -> {out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

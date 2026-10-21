"""pdfplumber 표를 '머리글 한 줄 + 병합 칸 채운 본문' 표로 바꾼다.

1. 격자: 칸 사각형(table.cells)의 x·y 경계로 열·행 경계를 만든다.
2. 칸 펼치기: 칸 사각형마다 덮는 행·열 범위를 계산해 그 범위 전체에 같은 글자를 채운다.
   - 가로 병합('은행'이 금액·비중 두 열에 걸침) → 두 열 모두 '은행'
   - 세로 병합('구 분'이 머리글 두 줄에 걸침, 행 머리가 여러 행에 걸침) → 걸친 행 모두 같은 값
   - 어느 칸 사각형에도 안 들어간 단어(테두리가 빠진 칸)는 위치로 격자 칸에 넣는다.
3. 쌓인 행 나누기: 행 사이 가로선이 없어 한 칸에 여러 줄이 들어 있으면 줄 위치로 행을 나눈다.
4. 머리글: 첫 행 + 숫자 없는 짧은 행(본문 숫자 행 앞까지, 최대 3줄)을 머리글로 보고,
   열마다 위→아래로 이어 한 줄로 만든다('은행' + '금액' → '은행 금액'). 왼쪽 위 빈칸은 '구분'.
5. 색칠 범위 표: fills(색 채움 사각형)를 주면 글자 없는 칸의 색칠을 보고, 본문이 색칠·빈칸뿐인 표는
   행마다 이어진 색칠 칸을 머리글 범위로 바꾼다('30년형 | 55~57'). kind='range'.
6. 검사: 열 이름 중복·빈 열 이름·빠진 글자·행 구분 미검출 등이 있으면 problems에 남긴다(호출 쪽에서 비전으로 다시 읽음).
"""
import re
from dataclasses import dataclass, field
from typing import Callable

HANGUL_RE = re.compile(r"[가-힣]")
MARK = "■"   # 색칠 칸 표시(색칠 범위 표로 바꿀 때만 쓰고, 아니면 지움)
LETTER_SPACED_RE = re.compile(r"^(?:[가-힣] )+[가-힣]$")   # '구 분', '합 계' 같은 자간 띄운 머리글


@dataclass
class TableResult:
    header: list[str]
    rows: list[list[str]]
    n_header_rows: int
    problems: list[str] = field(default_factory=list)
    kind: str = "grid"   # grid | range(색칠 칸을 범위로 바꾼 표)

    def to_md(self) -> str:
        esc = lambda s: s.replace("|", "\\|")
        lines = ["| " + " | ".join(map(esc, self.header)) + " |", "|" + "---|" * len(self.header)]
        lines += ["| " + " | ".join(map(esc, r)) + " |" for r in self.rows]
        return "\n".join(lines)


def is_number(s: str) -> bool:
    return bool(s) and bool(re.search(r"\d", s)) and not HANGUL_RE.search(s)


def is_value(s: str) -> bool:
    """숫자이거나 값 없음 표시(-)."""
    return is_number(s) or s.strip() in {"-", "–", "—", "…", "·"}


def _edges(values: list[float], tol: float = 2.0) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _split_at_edges(words: list[dict], xs: list[float]) -> list[dict]:
    """열 경계선이 단어 안을 지나가면 그 자리에서 끊는다(옆 칸 숫자끼리 붙은 '16.1594,743' → '16.1', '594,743').

    숫자 단어만 끊는다(여러 열에 걸친 병합 칸의 글자 '부동산집합투자증권'은 끊지 않음).
    글자 좌표('chars')가 있는 단어만 끊는다(PyMuPDF 경로). 없으면 그대로 둔다.
    """
    out = []
    for w in words:
        chars = w.get("chars")
        cuts = [x for x in xs if w["x0"] + 1 < x < w["x1"] - 1] if chars and is_number(w["text"]) else []
        if not cuts:
            out.append(w)
            continue
        part: list = []
        for ch in chars:
            if part and any(part[-1][1] - 1 <= x <= ch[0] + 1 for x in cuts):
                out.append(_word_from(part, w))
                part = []
            part.append(ch)
        out.append(_word_from(part, w))
    return out


def _word_from(chars: list, w: dict) -> dict:
    return {"x0": chars[0][0], "x1": chars[-1][1], "top": w["top"], "bottom": w["bottom"],
            "text": "".join(c[2] for c in chars), "chars": chars}


def _index(edges: list[float], v: float) -> int:
    return min(range(len(edges)), key=lambda i: abs(edges[i] - v))


def _lines(words: list[dict], tol: float = 3.0) -> list[tuple[float, str, list[dict]]]:
    """단어를 줄로 묶는다 → [(줄 위치, 줄 글자, 단어들)]"""
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    out = []
    for r in rows:
        r.sort(key=lambda w: w["x0"])
        out.append((r[0]["top"], " ".join(w["text"] for w in r), r))
    return out


def _inside(w: dict, box) -> bool:
    cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
    return box[0] - 0.5 <= cx <= box[2] + 0.5 and box[1] - 0.5 <= cy <= box[3] + 0.5


def _norm_cell(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "") if LETTER_SPACED_RE.match(s) else s


def _covered(cell, fills: list[tuple]) -> float:
    """칸 너비 중 색 막대가 덮는 비율.

    막대는 칸 높이보다 얇게 그려지는 경우가 많아(간트형 막대) 면적 대신,
    세로 중심이 칸 안에 있고 칸 높이의 25% 이상인 색 사각형이 가로로 얼마나 덮는지 본다.
    """
    cw, ch = max(1e-6, cell[2] - cell[0]), cell[3] - cell[1]
    s = 0.0
    for f in fills:
        cy = (f[1] + f[3]) / 2
        if not (cell[1] <= cy <= cell[3]) or (f[3] - f[1]) < 0.25 * ch:
            continue
        w = min(cell[2], f[2]) - max(cell[0], f[0])
        if w > 0:
            s += w
    return min(1.0, s / cw)


def extract_table(page, table, join: Callable[[str, str], str], fills: list[tuple] | None = None) -> TableResult:
    """fills: 색 있는 채움 사각형 [(x0, y0, x1, y1)] — 주면 색칠 칸 표(간트형)를 범위 표로 바꾼다."""
    xs = _edges([c[0] for c in table.cells] + [c[2] for c in table.cells])
    ys = _edges([c[1] for c in table.cells] + [c[3] for c in table.cells])
    n_rows, n_cols = len(ys) - 1, len(xs) - 1
    # within_bbox는 표 테두리에 조금이라도 걸친 단어(마지막 행 등)를 버리므로 단어 중심으로 고른다
    words = _split_at_edges([w for w in page.extract_words() if _inside(w, table.bbox)], xs)

    # ---- 칸 모으기: (r0, r1, c0, c1, 줄 목록) -------------------------------
    cells = []
    used: set[int] = set()
    for box in table.cells:
        c0, c1 = _index(xs, box[0]), _index(xs, box[2])
        r0, r1 = _index(ys, box[1]), _index(ys, box[3])
        if c1 <= c0 or r1 <= r0:
            continue
        idx = [i for i, w in enumerate(words) if i not in used and _inside(w, box)]
        used.update(idx)
        cells.append([r0, r1, c0, c1, _lines([words[i] for i in idx])])
    # 테두리가 빠져 어느 칸에도 안 들어간 단어는 위치로 격자 칸에 넣는다
    loose: dict[tuple[int, int], list[dict]] = {}
    for i, w in enumerate(words):
        if i in used:
            continue
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        c = max(0, min(n_cols - 1, sum(1 for x in xs if x <= cx) - 1))
        r = max(0, min(n_rows - 1, sum(1 for y in ys if y <= cy) - 1))
        loose.setdefault((r, c), []).append(w)
    for (r, c), ws in loose.items():
        cells.append([r, r + 1, c, c + 1, _lines(ws)])

    # ---- 쌓인 행: 한 격자 행 안의 줄 수만큼 행을 나눈다 ----------------------
    anchors: list[list[float]] = []
    for r in range(n_rows):
        single = [c for c in cells if c[0] == r and c[1] == r + 1 and c[4]]
        # 행 기준은 값(숫자) 칸의 줄 위치로 잡는다. 세로쓰기 행 머리(한 줄에 한 글자)는 기준에서 뺀다.
        # 값 칸이 없는 글자 행(비교표의 긴 설명 칸)은 칸 안 줄바꿈일 뿐이므로 나누지 않는다.
        valued = [c for c in single if all(is_value(w["text"]) for _, _, ws in c[4] for w in ws)]
        tops = max(([t for t, _, _ in c[4]] for c in valued), key=len, default=[])
        anchors.append(tops if len(tops) > 1 else [])
    sub = [max(1, len(a)) for a in anchors]
    base = [sum(sub[:r]) for r in range(n_rows)]
    total = sum(sub)
    grid = [[""] * n_cols for _ in range(total)]

    def fill(rr: range, cc: range, text: str) -> None:
        for i in rr:
            for j in cc:
                grid[i][j] = text

    def col_of(w: dict) -> int:
        cx = (w["x0"] + w["x1"]) / 2
        return max(0, min(n_cols - 1, sum(1 for x in xs if x <= cx) - 1))

    def place(rr: range, c0: int, c1: int, text: str, ws: list[dict]) -> None:
        # 여러 열에 걸친 칸에 숫자가 열마다 하나씩 있으면(본문에 세로선이 없는 표) 열별로 나눈다.
        # 글자(머리글·행 머리)는 병합 칸으로 보고 걸친 열 모두에 채운다.
        # 숫자는 병합 칸이어도 걸친 열 모두에 복사하지 않고, 숫자가 놓인 열에만 넣는다(값 중복 방지).
        if c1 - c0 > 1 and ws and all(is_value(w["text"]) for w in ws):
            by_col: dict[int, list[str]] = {}
            for w in ws:
                by_col.setdefault(min(max(col_of(w), c0), c1 - 1), []).append(w["text"])
            for j, parts in by_col.items():
                fill(rr, range(j, j + 1), " ".join(parts))
            return
        fill(rr, range(c0, c1), text)

    for r0, r1, c0, c1, lines in cells:
        if not lines:
            continue
        texts = [t for _, t, _ in lines]
        # 세로쓰기 행 머리: 한 줄에 한두 글자씩 쌓인 글자('저'/'량')는 한 단어('저량')로 붙여 걸친 행 모두에 채운다
        if (len(lines) >= 2 and all(len(t) <= 2 and HANGUL_RE.search(t) for t in texts)
                and (r1 - r0 > 1 or sub[r0] > 1)):
            fill(range(base[r0], base[r1 - 1] + sub[r1 - 1]), range(c0, c1), "".join(texts))
            continue
        if r1 == r0 + 1 and sub[r0] > 1:
            k = sub[r0]
            row = lambda i: range(base[r0] + i, base[r0] + i + 1)
            if len(lines) == k:                                   # 줄마다 한 행
                for i, (_, t, ws) in enumerate(lines):
                    place(row(i), c0, c1, t, ws)
            elif all(is_value(w["text"]) for _, _, ws in lines for w in ws):  # 값은 가장 가까운 줄 위치의 행으로
                for top, t, ws in lines:
                    place(row(min(range(k), key=lambda i: abs(anchors[r0][i] - top))), c0, c1, t, ws)
            elif len(lines) > k:
                # 행보다 줄이 많은 글자 칸(상위 행 머리 '증감 내역'이 하위 행 머리 사이에 끼어 있는 칸)은
                # 줄마다 가장 가까운 행에 붙인다
                per: dict[int, list[str]] = {}
                for top, t, _ in lines:
                    per.setdefault(min(range(k), key=lambda i: abs(anchors[r0][i] - top)), []).append(t)
                for i, ts in per.items():
                    fill(row(i), range(c0, c1), _join_lines(ts, join))
            else:                                                 # 줄이 적은 글자 칸은 세로 병합된 행 머리
                fill(range(base[r0], base[r0] + k), range(c0, c1), _join_lines(texts, join))
        else:
            rr = range(base[r0], base[r1 - 1] + sub[r1 - 1])
            if r1 - r0 > 1 and len(lines) > 1:
                # 여러 행에 걸친 칸: 줄마다 들어갈 행(쌓인 행이면 가장 가까운 줄 위치)을 찾는다.
                # 모든 행에 줄이 하나 이상 있으면 행 사이 테두리만 빠진 칸 → 행별로 넣는다.
                # 줄이 없는 행이 있으면 진짜 세로 병합(가운데 쓴 행 머리) → 걸친 행 모두 같은 값.
                per: dict[int, list] = {}
                for top, t, ws in lines:
                    r = max(r0, min(r1 - 1, sum(1 for y in ys if y <= top + 2) - 1))
                    i = min(range(sub[r]), key=lambda i: abs(anchors[r][i] - top)) if anchors[r] else 0
                    per.setdefault(base[r] + i, []).append((t, ws))
                if len(per) == len(rr):
                    for target, items in per.items():
                        if len(items) == 1:
                            place(range(target, target + 1), c0, c1, items[0][0], items[0][1])
                        else:
                            fill(range(target, target + 1), range(c0, c1), _join_lines([t for t, _ in items], join))
                else:
                    fill(rr, range(c0, c1), _join_lines(texts, join))
            elif len(lines) == 1:
                place(rr, c0, c1, texts[0], lines[0][2])
            else:
                fill(rr, range(c0, c1), _join_lines(texts, join))

    # 색칠 칸: 글자 없는 칸을 색 사각형이 절반 이상 덮으면 ■ (첫 행은 머리글이라 제외, 쌓인 행은 제외)
    if fills:
        for r in range(1, n_rows):
            if sub[r] != 1:
                continue
            for j in range(n_cols):
                cell = (xs[j], ys[r], xs[j + 1], ys[r + 1])
                if not grid[base[r]][j] and _covered(cell, fills) >= 0.5:
                    grid[base[r]][j] = MARK

    grid = [[_norm_cell(c) for c in row] for row in grid]
    grid = _drop_empty_and_duplicate_cols([r for r in grid if any(r)])

    # ---- 머리글 ----------------------------------------------------------
    numeric = lambda row: any(is_number(c) for c in row[1:])
    n_h = 1
    while n_h < min(3, len(grid) - 1):
        row = grid[n_h]
        later_numeric = any(numeric(r) for r in grid[n_h + 1:])
        if numeric(row) or not later_numeric or any(len(c) > 12 for c in row if c):
            break
        n_h += 1
    header = []
    for j in range(len(grid[0])):
        parts: list[str] = []
        for r in range(n_h):
            if grid[r][j] and grid[r][j] not in parts:
                parts.append(grid[r][j])
        header.append(" ".join(parts) or ("구분" if j == 0 else ""))
    header, body = _merge_label_columns(header, grid[n_h:])

    # ---- 색칠 범위 표 --------------------------------------------------------
    # 본문(행 머리 열 제외)이 ■와 빈칸뿐이면 칸 색칠로 범위를 나타낸 표(간트형)다.
    # 행마다 이어진 ■ 묶음을 머리글 값 범위로 바꾼다: 55·56·57 칸 → '55~57'.
    values = [c for r in body for c in r[1:]]
    if values and all(c in ("", MARK) for c in values) and values.count(MARK) >= max(1, 0.1 * len(values)):
        rows = []
        for r in body:
            runs, start = [], None
            for j in range(1, len(r) + 1):
                on = j < len(r) and r[j] == MARK
                if on and start is None:
                    start = j
                elif not on and start is not None:
                    a, b = header[start], header[j - 1]
                    runs.append(a if a == b else f"{a}~{b}")
                    start = None
            rows.append([r[0], ", ".join(runs)])
        return TableResult([header[0], "해당 범위"], rows, n_h, [], kind="range")
    if MARK in values or any(MARK in h for h in header):  # 색칠 표가 아니면 표시를 지우고 원래대로
        body = [[("" if c == MARK else c) for c in r] for r in body]
        grid = [[("" if c == MARK else c) for c in r] for r in grid]
        header = [(" ".join(p for p in h.split() if p != MARK) or ("구분" if j == 0 else "")) for j, h in enumerate(header)]

    # ---- 검사 ------------------------------------------------------------
    problems = []
    named = [h for h in header if h]
    if len(named) != len(set(named)):
        dup = sorted({h for h in named if named.count(h) > 1})
        problems.append(f"열 이름 중복: {dup}")
    if any(not h for h in header):
        problems.append("빈 열 이름")
    cell_text = " ".join(c for row in grid for c in row).replace(" ", "")
    missing = [w["text"] for w in words if w["text"].replace(" ", "") not in cell_text]
    if missing:
        problems.append(f"빠진 글자: {missing[:5]}")
    # 행 구분선이 점선 등이라 검출되지 않아 본문 전체가 한두 칸에 몰린 표
    tall = {c[2] for c in cells if len(c[4]) >= 4}
    if len(body) <= 2 and len(tall) >= 2:
        problems.append("행 구분 미검출(여러 줄이 한 칸에 몰림)")
    if not body:
        problems.append("본문 행 없음")
    elif any(is_value(c) for r in body for c in r[1:]):
        empty = [r[0] for r in body if r[0] and not any(r[1:])]
        if empty:
            problems.append(f"값 없는 행: {empty[:3]}")
    return TableResult(header, body, n_h, problems)


def _merge_label_columns(header: list[str], body: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """같은 머리글('구분')이 걸친 왼쪽 행 머리 열 여러 개를 한 열로 합친다.

    예: 구분 | 구분 | DB …  →  구분 | DB …,  본문은 '은행 신한은행'처럼 상위·하위 행 머리를 이어 쓴다.
    """
    j = 0
    while j + 1 < len(header) and header[j] == header[j + 1] and not any(is_value(r[j + 1]) for r in body if r[j + 1]):
        for r in body:
            a, b = r[j], r[j + 1]
            r[j] = b if not a or a == b else a if not b else f"{a} {b}"
            del r[j + 1]
        del header[j + 1]
    return header, body


def _join_lines(texts: list[str], join: Callable[[str, str], str]) -> str:
    out = ""
    for t in texts:
        out = join(out, t) if out else t
    return out


def _drop_empty_and_duplicate_cols(grid: list[list[str]]) -> list[list[str]]:
    """빈 열과, 바로 오른쪽 열과 겹치는 군더더기 열(격자 경계가 하나 더 생겨 갈라진 열)을 뺀다."""
    if not grid:
        return grid
    n = len(grid[0])
    keep = []
    for j in range(n):
        col = [r[j] for r in grid]
        if not any(col):
            continue
        if j + 1 < n and all(not a or a == r[j + 1] for a, r in zip(col, grid)):
            continue  # 이 열의 글자가 모두 오른쪽 열에도 있다
        keep.append(j)
    return [[r[j] for j in keep] for r in grid]

"""임베딩 모델을 바꿔 다시 돌린 실험(results_emb2/)을 원본(results/)과 나란히 비교하는 표를 만든다.

검색 단계: 실험 이름(B0-base, E1-a-extended, ...)이 같은 실행끼리 hit@5 등을 비교한다(같은 이름이 여럿이면 가장 나중 것).
답변 단계: 설정별(반복 평균)로 pass·핵심 항목 포함 등을 비교하고, 각 임베딩 안에서 기준(B0) 대비 차이도 낸다.

사용법:
    python -m rag.compare_embeddings --md docs/embedding_comparison_tables.md
    (다른 폴더를 비교하려면 --a results --b results_emb2)
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .config import ROOT
from .report import LABEL, NAME_RE, ORDER

RET_RE = re.compile(r"^\d{8}_\d{6}_retrieval_((?:B0|E\d)(?:-[a-c])?-(?:base|extended))-(?:similarity|bm25|hybrid|mmr)$")
RET_ORDER = ["B0", "E1-a", "E1-b", "E2-a", "E2-b", "E3-a", "E3-b", "E4-a", "E4-b", "E5-a", "E5-b", "E6-c"]
RET_LABEL = {"B0": "B0 Recursive 500/50", "E1-a": "E1-a 크기 300", "E1-b": "E1-b 크기 800", "E2-a": "E2-a 겹침 0",
             "E2-b": "E2-b 겹침 100", "E3-a": "E3-a 구조 단위", "E3-b": "E3-b 구조 + parent-child",
             "E4-a": "E4-a 제목 접두어", "E4-b": "E4-b 표 보존", "E5-a": "E5-a BM25", "E5-b": "E5-b hybrid", "E6-c": "E6-c MMR"}
SETS = {"base": ("기본 세트 (용어집+약관)", "base"), "ext": ("확장 세트 (가이드북+추가 문서)", "extended")}


def load(path: Path):
    return json.loads((path).read_text(encoding="utf-8"))


def collect_retrieval(runs: Path) -> dict:
    """{이름: summary all/all}, 같은 이름은 가장 나중 실행"""
    out = {}
    for d in sorted(runs.glob("*")):
        m = RET_RE.match(d.name)
        if m and (d / "summary.json").exists():
            out[m.group(1)] = load(d / "summary.json")["groups"]["all"]["all"]
    return out


def collect_answers(runs: Path) -> dict:
    """{세트: {설정: [all/all 요약, ...]}}"""
    out: dict = defaultdict(lambda: defaultdict(list))
    for d in sorted(runs.glob("*")):
        m = NAME_RE.match(d.name)
        if not m or not (d / "summary.json").exists():
            continue
        name = re.sub(r"-(r2|rep\d+)$", "", m.group(2))
        setting, _, which = name.rpartition("-")
        if which in ("base", "ext"):
            out[which][setting].append(load(d / "summary.json")["groups"]["all"]["all"])
    return out


def mean(runs: list, key: str):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def cell(v, pct=False, digits=3):
    return "-" if v is None else (f"{v * 100:.1f}" if pct else f"{v:.{digits}f}")


def delta(a, b, pct=False, digits=3):
    if a is None or b is None:
        return "-"
    d = (b - a) * (100 if pct else 1)
    return f"{d:+.1f}" if pct else f"{d:+.{digits}f}"


def retrieval_table(which: str, a: dict, b: dict, la: str, lb: str) -> str:
    title, suffix = SETS[which]
    lines = [f"### 검색 단계: {title}", "",
             f"| 실험 | hit@5 {la} | hit@5 {lb} | Δ | recall@5 {la} | recall@5 {lb} | Δ | source_coverage@5 {la} | {lb} | Δ "
             f"| mrr {la} | {lb} | Δ | 컨텍스트 글자 {la} | {lb} |", "|" + "---|" * 15]
    for key in RET_ORDER:
        name = f"{key}-{suffix}"
        x, y = a.get(name), b.get(name)
        if not x and not y:
            continue
        x, y = x or {}, y or {}
        row = [RET_LABEL[key]]
        for m, digits in (("hit@5", 3), ("recall@5", 3), ("source_coverage@5", 3), ("mrr", 3)):
            row += [cell(x.get(m)), cell(y.get(m)), delta(x.get(m), y.get(m))]
        row += [f"{x.get('context_chars@5', 0):,.0f}", f"{y.get('context_chars@5', 0):,.0f}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def answer_table(which: str, a: dict, b: dict, la: str, lb: str) -> str:
    title, _ = SETS[which]
    A, B = a.get(which, {}), b.get(which, {})
    lines = [f"### 답변 단계: {title}", "",
             "실행 열은 평가 횟수(001 / 002). 값은 반복 평균이다. 'B0 대비'는 각 임베딩 안에서 B0와의 차이(%p)다.", "",
             f"| 설정 | 실행 | pass % {la} | pass % {lb} | Δ | 포함 % {la} | 포함 % {lb} | Δ | "
             f"pass B0 대비 {la} | {lb} | 포함 B0 대비 {la} | {lb} | 근거 준수 % {la} | {lb} | 입력 토큰 {la} | {lb} |",
             "|" + "---|" * 16]
    for key in ORDER:
        x, y = A.get(key), B.get(key)
        if not x and not y:
            continue
        x, y = x or [], y or []
        row = [LABEL[key], f"{len(x)} / {len(y)}"]
        for m in ("pass", "must_include_rate"):
            mx, my = mean(x, m), mean(y, m)
            row += [cell(mx, True), cell(my, True), delta(mx, my, True)]
        for m in ("pass", "must_include_rate"):
            bx, by = mean(A.get("B0", []), m), mean(B.get("B0", []), m)
            mx, my = mean(x, m), mean(y, m)
            row += [delta(bx, mx, True) if key != "B0" else "", delta(by, my, True) if key != "B0" else ""]
        row += [cell(mean(x, "faithfulness"), True), cell(mean(y, "faithfulness"), True),
                f"{mean(x, 'input_tokens_mean') or 0:,.0f}", f"{mean(y, 'input_tokens_mean') or 0:,.0f}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="results", help="원본 결과 폴더(rag_mini_project 기준)")
    ap.add_argument("--b", default="results_emb2", help="비교할 결과 폴더")
    ap.add_argument("--label-a", default="emb-001")
    ap.add_argument("--label-b", default="emb-2")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    ra, rb = ROOT / args.a / "runs", ROOT / args.b / "runs"
    parts = []
    for which in ("base", "ext"):
        parts.append(retrieval_table(which, collect_retrieval(ra), collect_retrieval(rb), args.label_a, args.label_b))
    aa, ab = collect_answers(ra), collect_answers(rb)
    for which in ("base", "ext"):
        parts.append(answer_table(which, aa, ab, args.label_a, args.label_b))
    text = "\n\n".join(parts)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

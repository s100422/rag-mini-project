"""답변 단계(rag·agent) 실험을 설정별로 묶어 평균과 범위를 보여 준다.

같은 설정을 여러 번 돌린 실행(이름 끝의 -r2, -rep2 등)은 한 설정으로 묶는다. 같은 설정을 두 번 돌려도 pass가 3~6%p 달라지므로
(docs/experiment_results.md) 설정 하나의 값은 평균과 [최소–최대]로 본다.

사용법:
    python -m rag.report                 # 표를 화면에 출력
    python -m rag.report --md out.md     # 마크다운 파일로도 저장
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict

from .experiment import RUNS

NAME_RE = re.compile(r"^\d{8}_\d{6}_(rag|agent)_(.+?)-similarity$")
ORDER = ["B0", "E3-a", "E3-b", "E4-a", "E4-b", "E6-a", "E6-b", "E8a", "E8b", "E7", "E7a", "E7b"]
LABEL = {"B0": "B0 기준(Recursive 500/50, k=5)", "E3-a": "E3-a 구조 단위", "E3-b": "E3-b 구조 + parent-child",
         "E4-a": "E4-a 제목 접두어", "E4-b": "E4-b 표 보존", "E6-a": "E6-a k=3", "E6-b": "E6-b k=10",
         "E8a": "E8a 합침: E3-b + k=3", "E8b": "E8b 합침: E3-b + k=10", "E7": "E7 Agent 전체", "E7a": "E7a Agent: 재검색 끔", "E7b": "E7b Agent: 질문 분석·분기 끔"}
KEYS = ["pass", "must_include_rate", "coarse_match", "faithfulness", "citation_precision", "citation_recall",
        "unsupported_claims", "blocked"]


def collect() -> dict:
    """{세트: {설정: [실행 요약, ...]}}"""
    out: dict = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS.glob("*")):
        m = NAME_RE.match(d.name)
        if not m or not (d / "summary.json").exists():
            continue
        name = re.sub(r"-(r2|rep\d+)$", "", m.group(2))
        setting, _, which = name.rpartition("-")
        if which not in ("base", "ext"):
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))["groups"]
        a = s["all"]["all"]
        k = cfg["settings"]["retrieval"]["k"]
        out[which][setting].append({"id": d.name, "a": a, "k": k, "groups": s, "n_errors": a["n_errors"]})
    return out


def agg(runs: list[dict], key: str):
    vals = [r["a"][key] for r in runs if r["a"].get(key) is not None]
    return (statistics.mean(vals), min(vals), max(vals)) if vals else None


def fmt(t, pct=False, digits=3) -> str:
    if t is None:
        return "-"
    m, lo, hi = t
    if pct:
        return f"{m * 100:.1f} [{lo * 100:.0f}–{hi * 100:.0f}]" if lo != hi else f"{m * 100:.1f}"
    return f"{m:.{digits}f}"


def table(which: str, settings: dict) -> str:
    title = {"base": "기본 세트 (용어집+약관, 100문항)", "ext": "확장 세트 (가이드북+추가 문서, 80문항)"}[which]
    head = ("| 설정 | 실행 | pass % [범위] | 핵심 항목 포함 % [범위] | 답변·보류 일치 % | 근거 준수 % | 인용 정밀도 / 재현율 % "
            "| 입력 토큰/문항 | 출력 토큰 | LLM 호출 | 응답 p50(초) | 오류·차단 |")
    lines = [f"### {title}", "", head, "|" + "---|" * 12]
    base = settings.get("B0")
    for key in ORDER:
        runs = settings.get(key)
        if not runs:
            continue
        g = lambda k: agg(runs, k)
        tok_in = statistics.mean(r["a"]["input_tokens_mean"] for r in runs)
        tok_out = statistics.mean(r["a"]["output_tokens_mean"] + r["a"]["thinking_tokens_mean"] for r in runs)
        calls = statistics.mean(r["a"]["llm_calls_mean"] for r in runs)
        p50 = statistics.mean(r["a"].get("total_ms_p50") or 0 for r in runs) / 1000
        err = sum(r["n_errors"] for r in runs)
        blk = sum(r["a"].get("n_blocked", 0) for r in runs)
        lines.append(f"| {LABEL[key]} | {len(runs)} | {fmt(g('pass'), True)} | {fmt(g('must_include_rate'), True)} | "
                     f"{fmt(g('coarse_match'), True)} | {fmt(g('faithfulness'), True)} | "
                     f"{fmt(g('citation_precision'), True)} / {fmt(g('citation_recall'), True)} | "
                     f"{tok_in:,.0f} | {tok_out:,.0f} | {calls:.2f} | {p50:.1f} | {err}·{blk} |")
    if base:
        lines += ["", "기준(B0) 대비 차이(%p, 평균끼리): " + ", ".join(
            f"{k} pass {(agg(r, 'pass')[0] - agg(base, 'pass')[0]) * 100:+.1f} / 정확도 "
            f"{(agg(r, 'must_include_rate')[0] - agg(base, 'must_include_rate')[0]) * 100:+.1f}"
            for k, r in settings.items() if k != "B0" and k in ORDER)]
    return "\n".join(lines)


def run_index(data: dict) -> str:
    """설정별 실험 ID 목록(결과를 실험 기록에서 찾아가는 용도)."""
    lines = ["| 세트 | 설정 | 실험 ID |", "|---|---|---|"]
    for which in ("base", "ext"):
        for key in ORDER:
            runs = data.get(which, {}).get(key)
            if runs:
                lines.append(f"| {which} | {key} | " + "<br>".join(f"`{r['id']}`" for r in runs) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="")
    ap.add_argument("--runs", action="store_true", help="설정별 실험 ID 목록을 출력")
    args = ap.parse_args()
    data = collect()
    if args.runs:
        print(run_index(data))
        return
    text = "\n\n".join(table(w, data[w]) for w in ("base", "ext") if w in data)
    print(text)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

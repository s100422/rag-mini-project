"""기준 실험 대비 비교 실험의 결과를 표로 본다(docs/experiment_plan.md 4.3절 해석 규칙).

- 대표 지표의 기준 값, 실험 값, 차이
- 문항별로 기준보다 나아진/나빠진 문항 수
- 그룹별 차이
- 답변 단계(rag·agent)는 사용량(토큰·호출·시간)과 '품질 1%p당 추가 자원'도 보여 준다

사용법:
    python -m rag.compare <기준 실험 ID> <실험 ID> [<실험 ID> ...] [--group question_type]
    python -m rag.compare --baseline-of <실험 ID> ...      # config의 baseline_id를 기준으로
"""
import argparse
import json
import sys

from .experiment import RUNS

RETRIEVAL_METRICS = ["hit", "recall", "all_hit", "precision", "source_coverage", "dup_rate", "context_chars", "mrr"]
ANSWER_METRICS = ["pass", "must_include_rate", "forbidden_violation", "faithfulness", "unsupported_claims",
                  "behavior_match", "coarse_match", "citation_precision", "citation_recall"]
USAGE = ["llm_calls_mean", "retrieval_calls_mean", "input_tokens_mean", "output_tokens_mean",
         "total_ms_p50", "total_ms_p95"]
HIGHER_IS_BETTER_SKIP = {"forbidden_violation", "unsupported_claims", "dup_rate", "context_chars"}


def load(exp_id: str) -> tuple[dict, dict, dict]:
    d = RUNS / exp_id
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))["groups"]
    cases = {r["case_id"]: r for r in map(json.loads, (d / "per_case.jsonl").read_text(encoding="utf-8").splitlines())}
    return cfg, summary, cases


def case_value(row: dict, stage: str, key: str, k: int):
    if stage == "retrieval":
        return row["metrics"][key]
    if key.endswith(f"@{k}"):
        return (row.get("retrieval_metrics") or {}).get(key)
    return (row.get("answer_metrics") or {}).get(key)


def wins_losses(base: dict, exp: dict, key: str, stage: str = "retrieval", k: int = 5) -> tuple[int, int]:
    up = down = 0
    for cid, b in base.items():
        if cid not in exp:
            continue
        x, y = case_value(b, stage, key, k), case_value(exp[cid], stage, key, k)
        if x is None or y is None:
            continue
        up += y - x > 1e-9
        down += y - x < -1e-9
    return up, down


def fmt(v) -> str:
    return "-" if v is None else f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="기준 실험 ID 다음에 비교할 실험 ID들 (--baseline-of면 실험 ID만)")
    ap.add_argument("--baseline-of", action="store_true", help="각 실험 config의 baseline_id를 기준으로 쓴다")
    ap.add_argument("--group", default="question_type", help="그룹별 차이를 볼 그룹(없으면 '')")
    args = ap.parse_args()

    pairs = ([(load(e)[0]["baseline_id"], e) for e in args.ids] if args.baseline_of
             else [(args.ids[0], e) for e in args.ids[1:]])
    for base_id, exp_id in pairs:
        bcfg, bsum, bcases = load(base_id)
        ecfg, esum, ecases = load(exp_id)
        stage, k = ecfg["stage"], ecfg["settings"]["retrieval"]["k"]
        if bcfg["settings"]["data"]["answer_files"] != ecfg["settings"]["data"]["answer_files"]:
            print(f"!! 질문 세트가 다릅니다: {base_id} vs {exp_id}")
        changes = "; ".join(f"{c['path']}: {c['baseline']}→{c['value']}" for c in ecfg.get("changes", []))
        print(f"\n## {exp_id}  ({stage})\n기준 {base_id} | 변경: {changes or '(기준 대비 기록 없음)'}")
        b, e = bsum["all"]["all"], esum["all"]["all"]
        print("지표".ljust(26) + "기준".rjust(11) + "실험".rjust(11) + "차이".rjust(11))
        keys = ([m if m == "mrr" else f"{m}@{k}" for m in RETRIEVAL_METRICS] if stage == "retrieval"
                else ANSWER_METRICS + [f"hit@{k}", f"source_coverage@{k}"] + USAGE)
        for key in keys:
            if b.get(key) is None or e.get(key) is None:
                continue
            print(key.ljust(26) + f"{fmt(b[key]):>11}{fmt(e[key]):>11}{e[key] - b[key]:>+11,.3f}")
        wl_keys = (["hit", "recall"] if stage == "retrieval" else ["pass", "must_include_rate"])
        for m in wl_keys:
            key = f"{m}@{k}" if stage == "retrieval" else m
            up, down = wins_losses(bcases, ecases, key, stage, k)
            print(f"{key} 문항: 나아짐 {up}, 나빠짐 {down}, 순 {up - down:+d}")
        if stage != "retrieval":
            dq = (e["pass"] - b["pass"]) * 100
            dtok = (e["input_tokens_mean"] + e["output_tokens_mean"]) - (b["input_tokens_mean"] + b["output_tokens_mean"])
            dcall = e["llm_calls_mean"] - b["llm_calls_mean"]
            dms = (e.get("total_ms_p50") or 0) - (b.get("total_ms_p50") or 0)
            print(f"자원: 문항당 토큰 {dtok:+,.0f}, LLM 호출 {dcall:+.2f}, 응답시간 p50 {dms:+,.0f}ms, pass {dq:+.1f}%p")
            if abs(dq) >= 0.5:
                print(f"  → pass 1%p당 추가 토큰 {dtok / dq:+,.0f}, 호출 {dcall / dq:+.3f}, 시간 {dms / dq:+,.0f}ms")
            else:
                print("  → pass 차이가 0.5%p 미만이라 1%p당 자원은 계산하지 않음")
        if args.group:
            for gname in dict.fromkeys([args.group, "expected_group"] if stage != "retrieval" else [args.group]):
                if gname not in esum:
                    continue
                m1, m2 = (f"hit@{k}", f"recall@{k}") if stage == "retrieval" else ("pass", "must_include_rate")
                print(f"그룹({gname}) {m1} / {m2}:")
                for g, x in esum[gname].items():
                    y = bsum.get(gname, {}).get(g)
                    if y and x.get(m1) is not None and y.get(m1) is not None:
                        print(f"  {g:<12} n={x['n']:<3} {m1} {y[m1]:.2f}→{x[m1]:.2f}  {m2} {y[m2]:.2f}→{x[m2]:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

"""pass에 실패한 문항을 가장 앞 단계 원인으로 분류한다(docs/experiment_plan.md 8절).

README의 '검색에서 근거를 놓친 경우'와 '근거를 찾았지만 답변을 잘못 쓴 경우'를 나눈다. 원인은 파이프라인 앞 단계부터 하나만 붙인다.
  blocked        모델 API가 프롬프트를 차단함
  retrieval_miss 정답 근거가 컨텍스트에 하나도 없음(hit@k = 0)
  behavior       답할지·보류할지(답변·정정 vs 되묻기·보류·거절)가 기대와 다름
  forbidden      금지 내용을 주장함
  partial_ctx    근거 일부만 찾았고(recall@k < 1) 핵심 항목이 빠짐
  answer_gap     정답 근거를 모두 찾았는데도 핵심 항목이 빠짐(답변 단계 문제)
정답 근거가 없는 negative 문항은 retrieval_miss·partial_ctx가 없다.

사용법: python -m rag.failure_analysis <실험 ID> [<실험 ID> ...] [--examples 3]
"""
import argparse
import json
import sys
from collections import Counter

from .experiment import RUNS

REASONS = ["blocked", "retrieval_miss", "behavior", "forbidden", "partial_ctx", "answer_gap"]


def classify(row: dict, k: int) -> str | None:
    m = row["answer_metrics"]
    if not m or m["pass"] == 1.0:
        return None
    ret = row.get("retrieval_metrics")
    if m.get("blocked"):
        return "blocked"
    if ret and ret[f"hit@{k}"] == 0:
        return "retrieval_miss"
    if m["coarse_match"] == 0:
        return "behavior"
    if m["forbidden_violation"]:
        return "forbidden"
    if ret and ret[f"recall@{k}"] < 1:
        return "partial_ctx"
    return "answer_gap"


def load(exp_id: str) -> tuple[dict, list[dict]]:
    d = RUNS / exp_id
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (d / "per_case.jsonl").read_text(encoding="utf-8").splitlines()]
    return cfg, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--examples", type=int, default=0, help="원인별 예시 문항 수")
    args = ap.parse_args()
    print("실험".ljust(44) + "문항".rjust(5) + "실패".rjust(5) + "".join(r.rjust(16) for r in REASONS))
    for exp_id in args.ids:
        cfg, rows = load(exp_id)
        k = cfg["settings"]["retrieval"]["k"]
        why = {r["case_id"]: classify(r, k) for r in rows}
        c = Counter(v for v in why.values() if v)
        print(exp_id.split("_", 2)[2][:43].ljust(44) + f"{len(rows):5d}{sum(c.values()):5d}" +
              "".join(f"{c[r]:16d}" for r in REASONS))
        for reason in REASONS if args.examples else []:
            ex = [r for r in rows if why[r["case_id"]] == reason][:args.examples]
            for r in ex:
                m = r["answer_metrics"]
                print(f"    [{reason}] {r['case_id']} 기대={r['expected_group']} 판정={m['behavior']} "
                      f"포함={m['must_include_rate']} | {r['question'][:50]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

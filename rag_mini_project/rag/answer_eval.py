"""기본 RAG·Agent의 답변을 평가하고 실험 기록(docs/experiment_records.md)에 저장한다.

문항마다 검색 → 답변 → 판정을 한 스레드에서 처리한다(문항끼리는 병렬).
평가 세트는 컬렉션이 색인한 문서로 정한다.
  - 용어집+약관 컬렉션: positive_cases + negative_cases (100문항, 근거 없는 negative 32개 포함)
  - 가이드북+추가 문서 컬렉션: positive_life_guide_cases + additional_cases (80문항)

사용법:
    python -m rag.answer_eval --name B0-base --collection base_rec_c500_o50
    python -m rag.answer_eval --name E3-b-base --collection base_pc_c500_o50 --baseline <기준 실험 ID>
    python -m rag.answer_eval --mode agent --name E7-base --collection base_rec_c500_o50 --baseline <RAG 실험 ID>
    python -m rag.answer_eval ... --limit 5            # 앞 5문항만(시험)
    RAG_RESULTS_DIR=<폴더> python -m rag.answer_eval ... # 정식 기록과 따로 저장
"""
import argparse
import os
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime

import langsmith as ls

from .config import EMBEDDING_MODEL, ROOT
from .evaluate import (ANSWERS, LABELS, GROUPS, _doc_group, _percentile, case_metrics, chunking_info,
                       data_files, load_labels, retrieval_settings, source_doc_of)
from .experiment import (RUNS, append_index, command_line, env_info, experiment_id, format_changes,
                         new_sweep_id, save_prompt, sha256_text, start_run, write_csv, write_json)
from .judge import (BEHAVIOR_GROUPS, MAX_OUTPUT_TOKENS as JUDGE_MAX_TOKENS, PROMPT as JUDGE_PROMPT, behavior_group,
                    blocked_metrics, judge_case, score_case)
from .llm import GEN_MODEL, JUDGE_MODEL
from .retrievers import DEFAULT_COLLECTION, Retriever
from . import rag

BASE_FILES = ["positive_cases.jsonl", "negative_cases.jsonl"]
EXT_FILES = ["positive_life_guide_cases.jsonl", "additional_cases.jsonl"]
ANSWER_KEYS = ["pass", "blocked", "must_include_rate", "must_include_all", "forbidden_violation", "faithfulness",
               "unsupported_claims", "citation_precision", "citation_recall", "cited_any", "behavior_match",
               "coarse_match"]
USAGE_KEYS = ["llm_calls", "retrieval_calls", "input_tokens", "output_tokens", "thinking_tokens"]
GROUP_KEYS = {**GROUPS, "expected_group": lambda c: c["expected_group"]}


# ---- 문항 -------------------------------------------------------------------

def answer_files_for(doc_types: list[str]) -> list[str]:
    if {"standard_terms", "glossary"} & set(doc_types):
        return BASE_FILES
    if {"life_guide", "additional_p4l"} & set(doc_types):
        return EXT_FILES
    raise ValueError(f"평가 세트를 정할 수 없습니다: {doc_types}")


def load_answer_cases(doc_types: list[str], indexed: set[str]) -> tuple[list[dict], list[str]]:
    labels = load_labels()
    files, cases = answer_files_for(doc_types), []
    for fname in files:
        for line in (ANSWERS / fname).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ids = r["expected_source_ids"]
            expected = [s for s in ids if source_doc_of(s) in indexed]
            cases.append({
                "case_id": r["case_id"], "question": r["question"], "set": r.get("set"),
                "category": r.get("category"), "difficulty": r.get("difficulty"),
                "expected_behavior": r["expected_behavior"], "expected_group": behavior_group(r["expected_behavior"]),
                "expected_source_ids": expected, "excluded_source_ids": [s for s in ids if s not in expected],
                "source_doc": "+".join(sorted({_doc_group(s) for s in expected})) or "none",
                **labels.get(r["case_id"], {"question_type": "n/a", "needs_table": False}),
                "must_include": r.get("must_include", []), "must_not_include": r.get("must_not_include", []),
                "reference_answer": r["reference_answer"], "answer_file": fname,
            })
    return cases, files


# ---- 실행 -------------------------------------------------------------------

def result_row(r, case: dict) -> dict:
    return {"rank": r.rank, "source_ids": r.source_ids, "parent_id": r.parent_id, "chunk_ids": r.matched_chunk_ids,
            "score": round(r.score, 6), "source_file": r.source_file, "heading": r.heading,
            "page_start": r.page_start, "page_end": r.page_end,
            "relevant": bool(set(r.source_ids) & set(case["expected_source_ids"])), "content": r.content}


CASE_TIMEOUT = 900  # 초. 한 문항이 이보다 오래 걸리면(호출이 응답 없이 걸린 경우) 오류로 기록하고 다음으로 넘어간다


def base_row(case: dict, ctx: dict) -> dict:
    return {"experiment_id": ctx["exp_id"], "sweep_id": ctx["sweep_id"],
            **{key: v for key, v in case.items() if key != "reference_answer"},
            "retrieved": [], "retrieval_metrics": None, "answer": None, "citations": [], "cited_sources": [],
            "usage": None, "latency": None, "steps": [], "trace": None, "judge": None, "answer_metrics": None,
            "error": None}


def process(case: dict, run_fn, retriever, search: dict, k: int, ctx: dict) -> dict:
    row = base_row(case, ctx)
    try:
        with ls.trace(name=f"{ctx['mode']}_question", run_type="chain", inputs={"question": case["question"]},
                      project_name=ctx["project"], tags=[ctx["exp_id"]],
                      metadata={"case_id": case["case_id"], "experiment_id": ctx["exp_id"],
                                "collection": ctx["collection"]}) as rt:
            rec = run_fn(case["question"], retriever, search)
            rt.end(outputs={"answer": rec.answer})
        row["trace"] = {"langsmith_project": ctx["project"], "langsmith_run_id": str(rt.id)}
        results = rec.results[:k]
        row["retrieved"] = [result_row(r, case) for r in results]
        if case["expected_source_ids"]:
            row["retrieval_metrics"] = case_metrics(results, case, [k])
        row["answer"], row["citations"] = rec.answer, rec.citations
        row["cited_sources"] = [{"n": n, "source_file": results[n - 1].source_file, "page_start": results[n - 1].page_start,
                                 "page_end": results[n - 1].page_end, "source_ids": results[n - 1].source_ids}
                                for n in dict.fromkeys(rec.citations) if 1 <= n <= len(results)]
        row["usage"] = rec.usage.to_dict()
        row["latency"] = {"retrieval_ms": round(rec.retrieval_ms, 1), "generation_ms": round(rec.generation_ms, 1),
                          "total_ms": round(rec.retrieval_ms + rec.generation_ms, 1)}
        row["steps"] = rec.steps
        blocked = getattr(rec, "blocked", "")
        if blocked:
            row["blocked"] = blocked
            row["answer_metrics"] = blocked_metrics(case)
        else:
            j = judge_case(case, results, rec.answer)
            row["judge"] = {"model_version": j.model_version, "latency_ms": round(j.latency_ms, 1),
                            "usage": {"input_tokens": j.input_tokens, "output_tokens": j.output_tokens,
                                      "thinking_tokens": j.thinking_tokens},
                            "output": j.parsed.model_dump()}
            row["answer_metrics"] = score_case(case, results, rec.citations, j.parsed)
    except Exception as e:  # noqa: BLE001  한 문항의 실패가 전체를 멈추지 않게 한다
        row["error"] = f"{type(e).__name__}: {e}"[:500]
    return row


# ---- 집계 -------------------------------------------------------------------

def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def aggregate(rows: list[dict], k: int) -> dict:
    ok = [r for r in rows if r["answer_metrics"]]
    out = {"n": len(rows), "n_errors": len(rows) - len(ok)}
    for key in ANSWER_KEYS:
        out[key] = _mean(r["answer_metrics"][key] for r in ok)
    ret = [r["retrieval_metrics"] for r in rows if r["retrieval_metrics"]]
    out["n_with_sources"] = len(ret)
    for name in ("hit", "recall", "precision", "source_coverage", "dup_rate", "context_chars"):
        out[f"{name}@{k}"] = _mean(m[f"{name}@{k}"] for m in ret)
    for key in USAGE_KEYS:
        vals = [r["usage"][key] for r in ok]
        out[f"{key}_mean"], out[f"{key}_total"] = _mean(vals), sum(vals)
    for part in ("retrieval_ms", "generation_ms", "total_ms"):
        vals = [r["latency"][part] for r in ok]
        if vals:
            out[f"{part}_p50"], out[f"{part}_p95"] = round(statistics.median(vals), 1), round(_percentile(vals, 95), 1)
    out["judge_incomplete"] = sum(r["answer_metrics"]["judge_incomplete"] for r in ok)
    out["n_blocked"] = sum(1 for r in ok if r["answer_metrics"]["blocked"])
    return out


def confusion(rows: list[dict]) -> dict:
    out: dict = {g: {h: 0 for h in BEHAVIOR_GROUPS} for g in BEHAVIOR_GROUPS}
    for r in rows:
        if r["answer_metrics"]:
            row = out[r["expected_group"]]
            row[r["answer_metrics"]["behavior"]] = row.get(r["answer_metrics"]["behavior"], 0) + 1
    return out


# ---- 실행 -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--note", default="")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--mode", default="rag", choices=["rag", "agent"])
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--method", default="similarity", choices=["similarity", "mmr", "bm25", "hybrid"])
    ap.add_argument("--k", type=int, default=5, help="컨텍스트에 넣는 결과 수")
    ap.add_argument("--fetch-k", type=int, default=40)
    ap.add_argument("--lambda-mult", type=float, default=0.5)
    ap.add_argument("--weight-vector", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="앞 N문항만(시험)")
    ap.add_argument("--cases", default="", help="이 문항들만(쉼표로 구분, 시험)")
    ap.add_argument("--agent-options", default="", help="Agent 요소 끄기(ablation): retry=0 등, 쉼표로 구분")
    args = ap.parse_args()

    retriever = Retriever(args.collection)
    chunking, stats, doc_types, source_doc_ids = chunking_info(retriever)
    cases, files = load_answer_cases(doc_types, set(source_doc_ids))
    if args.cases:
        cases = [c for c in cases if c["case_id"] in args.cases.split(",")]
    if args.limit:
        cases = cases[:args.limit]
    if args.method in ("bm25", "hybrid"):
        retriever._ensure_bm25()  # 스레드가 동시에 색인을 만들지 않게 미리 만든다
    search = dict(method=args.method, k=args.k, fetch_k=args.fetch_k, lambda_mult=args.lambda_mult,
                  weight_vector=args.weight_vector)

    if args.mode == "agent":
        from . import agent
        run_fn, gen_settings, prompts = agent.make_runner(args.agent_options), agent.settings(args.agent_options), agent.PROMPTS
    else:
        run_fn = rag.run_rag
        gen_settings = {"mode": "rag", "model": GEN_MODEL, "temperature": 0.0, "thinking_budget": rag.THINKING_BUDGET,
                        "max_output_tokens": rag.MAX_OUTPUT_TOKENS, "prompt_version": rag.PROMPT_VERSION,
                        "context_format": "[근거 N] 파일명 p.쪽 + 원문(제목 없음)",
                        "response_format": "JSON(answer, citations)"}
        prompts = {"generation_system": rag.SYSTEM_PROMPT}
    prompt_refs = {n: {"file": f"prompts/{n}.txt", "sha256": sha256_text(t)} for n, t in prompts.items()}
    judge_ref = {"file": "prompts/judge.txt", "sha256": sha256_text(JUDGE_PROMPT)}

    sweep_id = new_sweep_id(args.name)
    exp_id = experiment_id(sweep_id, args.mode, args.method)
    project = os.getenv("LANGSMITH_PROJECT")
    config = {
        "experiment_id": exp_id, "sweep_id": sweep_id, "stage": args.mode,
        "created_at": datetime.now().isoformat(timespec="seconds"), "command": command_line(), "note": args.note,
        "settings": {
            "data": {"answer_files": files, "case_filter": "컬렉션이 색인한 문서 세트의 전체 문항(근거 없는 negative 포함)",
                     "n_cases": len(cases), "n_cases_with_sources": sum(1 for c in cases if c["expected_source_ids"]),
                     "doc_types": doc_types, "source_doc_ids": source_doc_ids},
            "chunking": chunking, "embedding": {"model": EMBEDDING_MODEL},
            "retrieval": retrieval_settings(args.method, args, args.k,
                                            retriever.vs._collection.configuration_json.get("hnsw", {})),
            "generation": {**gen_settings, "prompts": prompt_refs},
            "eval": {
                "ks": [args.k], "relevance_unit": "결과의 source_id(조항·용어·PDF 쪽), 상위 k개 = 컨텍스트",
                "labels": LABELS.relative_to(ROOT).as_posix(),
                "judge": {"model": JUDGE_MODEL, "temperature": 0.0, "max_output_tokens": JUDGE_MAX_TOKENS,
                          "prompt": judge_ref, "sees_expected_behavior": False},
                "behavior_groups": BEHAVIOR_GROUPS,
                "pass": "핵심 항목을 모두 포함 AND 금지 내용 없음 AND 답변·보류 계열이 기대와 같음(coarse_match)",
                "citation": "인용 근거 번호를 검색 결과의 근거 ID로 바꿔 정답 근거와 비교",
            },
        },
        "collection_stats": stats,
        "env": env_info(data_files(files, doc_types)),
    }
    run_dir = start_run(config, args.baseline)
    for name, text in prompts.items():
        save_prompt(run_dir, name, text)
    save_prompt(run_dir, "judge", JUDGE_PROMPT)

    ctx = {"exp_id": exp_id, "sweep_id": sweep_id, "mode": args.mode, "collection": args.collection,
           "project": project or "default"}
    t0 = time.perf_counter()
    done = 0

    def task(c):
        nonlocal done
        row = process(c, run_fn, retriever, search, args.k, ctx)
        done += 1
        if done % 20 == 0 or done == len(cases):
            print(f"  {done}/{len(cases)}", flush=True)
        return row

    pool = ThreadPoolExecutor(args.workers)
    futures = [pool.submit(task, c) for c in cases]
    rows, timed_out = [], 0
    for c, f in zip(cases, futures):
        try:
            rows.append(f.result(timeout=CASE_TIMEOUT))
        except FutureTimeout:
            timed_out += 1
            rows.append(base_row(c, ctx) | {"error": f"TimeoutError: {CASE_TIMEOUT}초 안에 끝나지 않음"})
    pool.shutdown(wait=False, cancel_futures=True)
    wall = time.perf_counter() - t0
    with (run_dir / "per_case.jsonl").open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups, flat = {}, []
    for gname, key in GROUP_KEYS.items():
        for gval in sorted({key(r) for r in rows}, key=str):
            s = aggregate([r for r in rows if key(r) == gval], args.k)
            groups.setdefault(gname, {})[gval] = s
            flat.append({"experiment_id": exp_id, "group": gname, "value": gval, **s})
    judge_usage = {k: sum(r["judge"]["usage"][k] for r in rows if r["judge"]) for k in
                   ("input_tokens", "output_tokens", "thinking_tokens")}
    write_json(run_dir / "summary.json", {"experiment_id": exp_id, "groups": groups,
                                          "behavior_confusion": confusion(rows), "wall_seconds": round(wall, 1),
                                          "judge_usage_total": judge_usage})
    write_csv(run_dir / "summary.csv", flat)
    try:
        ls.Client().flush()
    except Exception:  # noqa: BLE001
        pass

    a, r = groups["all"]["all"], config["settings"]["retrieval"]
    ch = config["settings"]["chunking"]
    append_index({
        "experiment_id": exp_id, "sweep_id": sweep_id, "stage": args.mode, "created_at": config["created_at"],
        "baseline_id": config["baseline_id"] or "", "changes": format_changes(config["changes"]),
        "env_changes": len(config["env_changes"]), "note": args.note, "collection": args.collection,
        "strategy": ch["strategy"], "chunk_size": ch["chunk_size"], "chunk_overlap": ch["chunk_overlap"],
        "title_prefix": ch["title_prefix"], "table_aware": ch["table_aware"], "embedding_model": EMBEDDING_MODEL,
        "retrieval_method": args.method, "k": args.k, "fetch_k": args.fetch_k,
        "weight_vector": r.get("weight_vector", ""), "lambda_mult": r.get("lambda_mult", ""),
        "generation_model": GEN_MODEL, "judge_model": JUDGE_MODEL, "answer_files": " ".join(files),
        "n_cases": a["n"], "n_errors": a["n_errors"], "n_blocked": a["n_blocked"],
        "pass_rate": a["pass"], "answer_accuracy": a["must_include_rate"], "forbidden_rate": a["forbidden_violation"],
        "faithfulness": a["faithfulness"], "unsupported_claims_mean": a["unsupported_claims"],
        "behavior_match": a["behavior_match"], "coarse_match": a["coarse_match"],
        "citation_precision": a["citation_precision"], "citation_recall": a["citation_recall"],
        "ctx_hit": a[f"hit@{args.k}"], "ctx_recall": a[f"recall@{args.k}"], "ctx_source_coverage": a[f"source_coverage@{args.k}"],
        "llm_calls_mean": a["llm_calls_mean"], "retrieval_calls_mean": a["retrieval_calls_mean"],
        "input_tokens_mean": a["input_tokens_mean"], "output_tokens_mean": a["output_tokens_mean"],
        "thinking_tokens_mean": a["thinking_tokens_mean"], "input_tokens_total": a["input_tokens_total"],
        "output_tokens_total": a["output_tokens_total"], "total_ms_p50": a.get("total_ms_p50"),
        "total_ms_p95": a.get("total_ms_p95"),
    })
    print(f"done {exp_id}" + (f"  changes: {format_changes(config['changes'])}" if config["changes"] else ""))
    print(f"n={a['n']} errors={a['n_errors']} blocked={a['n_blocked']} pass={a['pass']} 정확도={a['must_include_rate']} 금지위반={a['forbidden_violation']} "
          f"근거준수={a['faithfulness']} 행동일치={a['behavior_match']}/{a['coarse_match']} "
          f"인용 P/R={a['citation_precision']}/{a['citation_recall']} hit@{args.k}={a[f'hit@{args.k}']} "
          f"토큰(in/out)={a['input_tokens_mean']}/{a['output_tokens_mean']} p50={a.get('total_ms_p50')}ms 소요 {wall:.0f}s")
    if timed_out:  # 걸린 스레드가 남아 있으면 종료가 막히므로 결과를 모두 쓴 뒤 바로 끝낸다
        print(f"제한 시간 초과 {timed_out}문항", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

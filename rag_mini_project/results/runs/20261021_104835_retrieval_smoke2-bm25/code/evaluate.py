"""골든셋으로 검색기를 평가하고, 설정·문항별 결과·요약 통계를 저장한다.

평가 대상: 03_answers/*.jsonl 중 expected_source_ids에 이 컬렉션이 색인한 문서의 근거가 있는 문항.
  - 컬렉션이 색인한 문서는 청크 메타데이터의 source_doc_id로 정한다
    (kofia-terms-123-1709, fsc-glossary, fss-life-guide, housing_pension_2026 …).
  - 색인 밖의 근거 ID는 채점에서 빼고 excluded_source_ids에 남긴다.

지표는 실제 컨텍스트에 넣는 상위 k개 결과로 계산한다.
  - parent-child 컬렉션(약관·용어집): 결과 하나 = parent 하나(조항·용어).
  - parent 없는 컬렉션(가이드북·추가 문서): 결과 하나 = 청크 하나. 같은 쪽 청크 2개면 k 두 자리를 차지한다.
  - 결과의 source_id(조항·용어·쪽 단위)를 정답 ID와 비교한다.

지표 (문항별 → 평균)
  hit@k      상위 k개에 정답이 하나라도 있으면 1
  recall@k   정답 중 상위 k개에 들어온 비율
  all_hit@k  정답이 모두 상위 k개에 있으면 1 (근거가 여러 개인 문항용)
  precision@k 상위 k개 중 정답 비율(자리 단위: 같은 정답 쪽 청크가 두 번 나오면 두 자리 모두 정답)
  mrr        첫 정답 순위의 역수 (k_max 안에 없으면 0)
  latency_ms 검색 1회 시간(질의 임베딩·Postgres parent 조회 포함)

분석 그룹: 전체·difficulty·set·category·근거 수·source_doc(근거 문서: 약관/용어집, 가이드북/추가 문서별)

저장(설계: docs/experiment_records.md): 검색기마다 실험 ID 하나, 한 번에 돌린 것은 sweep_id로 묶는다.
  results/runs/<experiment_id>/config.json     설정 전체(settings) + 기준 대비 변경(changes) + 재현 정보(env)
  results/runs/<experiment_id>/per_case.jsonl  문항별 검색 결과(청크 ID·컨텍스트 원문)와 지표
  results/runs/<experiment_id>/summary.json/csv 그룹별 통계
  results/runs/<experiment_id>/code/           실행 당시 rag/*.py 사본
  results/experiments.csv                      실험당 한 줄(실험 간 비교용)

사용법:
    python -m rag.evaluate --name base --collection base_c350_o50
    python -m rag.evaluate --name w07 --methods hybrid --weight-vector 0.7 \\
        --baseline 20261021_103000_retrieval_base-hybrid --note "벡터 쪽 가중치를 높이면 서술형 질문이 나아질 것"
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from .chunking import SEPARATORS
from .config import EMBEDDING_MODEL, MD_DIR, ROOT
from .experiment import (RUNS, append_index, command_line, env_info, experiment_id, format_changes,
                         new_sweep_id, start_run, write_csv, write_json)
from .retrievers import BM25_TAGS, DEFAULT_COLLECTION, RRF_K, Retriever
from .sources import source_texts
from . import pg_store

ANSWERS = ROOT / "03_answers"
LABELS = ROOT / "eval_labels" / "question_types.csv"
METHODS = ["similarity", "mmr", "bm25", "hybrid"]


# ---- 평가 데이터 -----------------------------------------------------------

def source_doc_of(sid: str) -> str:
    """근거 ID → 문서 ID (청크 메타데이터 source_doc_id와 같은 형식).

    housing_pension_2026-pdf-15 → housing_pension_2026, fsc-glossary-1697 → fsc-glossary,
    kofia-terms-123-1709-005 → kofia-terms-123-1709
    """
    return sid.split("-pdf-")[0] if "-pdf-" in sid else sid.rsplit("-", 1)[0]


def _doc_group(sid: str) -> str:
    """분석용 문서 묶음: 약관 15개는 하나(kofia-terms)로, 나머지는 문서 ID 그대로."""
    return "kofia-terms" if sid.startswith("kofia-terms-") else source_doc_of(sid)


def load_labels() -> dict[str, dict]:
    """질문 유형 라벨(eval_labels/question_types.csv, 직접 붙이고 검수한 것)."""
    with LABELS.open(newline="", encoding="utf-8-sig") as fp:
        return {r["case_id"]: {"question_type": r["question_type"], "needs_table": r["needs_table"] == "1"}
                for r in csv.DictReader(fp)}


def load_cases(indexed_docs: set[str]) -> tuple[list[dict], list[str]]:
    labels = load_labels()
    files, cases = [], []
    for f in sorted(ANSWERS.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        picked = [r for r in rows if any(source_doc_of(s) in indexed_docs for s in r["expected_source_ids"])]
        if picked:
            files.append(f.name)
        for r in picked:
            ids = r["expected_source_ids"]
            expected = [s for s in ids if source_doc_of(s) in indexed_docs]
            cases.append({
                "case_id": r["case_id"],
                "question": r["question"],
                "set": r.get("set"),
                "category": r.get("category"),
                "difficulty": r.get("difficulty"),
                "expected_behavior": r.get("expected_behavior"),
                "expected_source_ids": expected,
                "excluded_source_ids": [s for s in ids if s not in expected],
                "source_doc": "+".join(sorted({_doc_group(s) for s in expected})),
                **labels[r["case_id"]],
                "must_include": r.get("must_include", []),
                "answer_file": f.name,
            })
    return cases, files


# ---- 지표 ------------------------------------------------------------------

def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _coverage(source: str, contents: list[str]) -> float:
    """정답 근거 원문 중 컨텍스트에 들어간 비율(글자 기준).

    컨텍스트를 줄 단위로 나눠 원문에서 찾은 구간을 표시한다. 청크가 근거 경계를 넘어도(다른 근거와 섞여도)
    줄마다 맞추므로 셀 수 있다. 5자 미만 줄은 우연히 맞기 쉬워 뺀다.
    """
    src = _squash(source)
    if not src:
        return 0.0
    covered = bytearray(len(src))
    for line in {_squash(l) for c in contents for l in c.splitlines()}:
        if len(line) < 5:
            continue
        start = src.find(line)
        while start != -1:
            covered[start:start + len(line)] = b"\x01" * len(line)
            start = src.find(line, start + 1)
    return sum(covered) / len(src)


def case_metrics(results: list, case: dict, ks: list[int]) -> dict:
    """상위 k개 결과(=컨텍스트)로 문항 지표를 계산한다."""
    retrieved = [r.source_id for r in results]
    exp = set(case["expected_source_ids"])
    texts = source_texts()
    phrases = [p for p in case["must_include"]
               if any(_squash(p) in _squash(texts.get(s, "")) for s in exp)]  # 원문에 그대로 있는 구절만
    m: dict[str, float | None] = {}
    for k in ks:
        top, contents = retrieved[:k], [r.content for r in results[:k]]
        found = exp & set(top)
        m[f"hit@{k}"] = float(bool(found))
        m[f"recall@{k}"] = len(found) / len(exp)
        m[f"all_hit@{k}"] = float(found == exp)
        m[f"precision@{k}"] = sum(1 for s in top if s in exp) / k
        m[f"dup_rate@{k}"] = 1 - len(set(top)) / len(top) if top else 0.0
        m[f"context_chars@{k}"] = sum(len(c) for c in contents)
        m[f"source_coverage@{k}"] = statistics.mean(_coverage(texts.get(s, ""), contents) for s in exp)
        ctx = _squash("".join(contents))
        m[f"key_phrase_recall@{k}"] = (sum(_squash(p) in ctx for p in phrases) / len(phrases)) if phrases else None
    first = next((i for i, s in enumerate(retrieved, start=1) if s in exp), None)
    m["mrr"] = 1 / first if first else 0.0
    m["first_relevant_rank"] = first
    m["key_phrases_checked"] = len(phrases)
    return m


def aggregate(rows: list[dict], metric_keys: list[str]) -> dict:
    """문항 평균. 값이 없는 문항(None, 예: 원문 구절이 없는 key_phrase_recall)은 빼고 평균한다."""
    lat = [r["latency_ms"] for r in rows]
    out = {"n": len(rows)}
    for key in metric_keys:
        vals = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        out[key] = round(statistics.mean(vals), 4) if vals else None
    out["n_key_phrase_cases"] = sum(1 for r in rows if r["metrics"]["key_phrases_checked"])
    out |= {
        "latency_ms_mean": round(statistics.mean(lat), 1),
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(_percentile(lat, 95), 1),
        "latency_ms_max": round(max(lat), 1),
    }
    return out


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    i = (len(xs) - 1) * p / 100
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


GROUPS = {
    "all": lambda c: "all",
    "difficulty": lambda c: c["difficulty"],
    "set": lambda c: c["set"],
    "category": lambda c: c["category"],
    "n_sources": lambda c: "1" if len(c["expected_source_ids"]) == 1 else "2+",
    "source_doc": lambda c: c["source_doc"],
    "question_type": lambda c: c["question_type"],   # term(용어형) / situation(상황형)
    "needs_table": lambda c: "table" if c["needs_table"] else "text",
}


# ---- 청킹 전략 기록 ----------------------------------------------------------

def chunking_info(retriever: Retriever) -> dict:
    data = retriever.vs.get(include=["documents", "metadatas"])
    metas, docs = data["metadatas"], data["documents"]
    configs = sorted({m.get("chunk_config") for m in metas})
    size = overlap = None
    if len(configs) == 1 and (m := re.fullmatch(r"c(\d+)_o(\d+)", configs[0] or "")):
        size, overlap = int(m[1]), int(m[2])
    lens = [len(d) for d in docs]
    common = {
        "collection": retriever.collection,
        "doc_types": sorted({m["doc_type"] for m in metas}),
        "source_doc_ids": sorted({m["source_doc_id"] for m in metas if m.get("source_doc_id")}),
        "child_splitter": "RecursiveCharacterTextSplitter",
        "chunk_config": configs[0] if len(configs) == 1 else configs,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "separators": SEPARATORS,
        "n_chunks": len(docs),
        "chunk_chars_mean": round(statistics.mean(lens), 1),
        "chunk_chars_median": statistics.median(lens),
        "chunk_chars_max": max(lens),
    }
    if not any("parent_id" in m for m in metas):
        pages = {(m["doc_id"], m["page_start"]) for m in metas}
        return {**common, "strategy": "flat", "split_unit": "PDF 쪽(청크가 쪽을 넘지 않음)",
                "n_pages": len(pages), "chunks_per_page_mean": round(len(docs) / len(pages), 2)}
    per_parent: dict[str, int] = {}
    for m in metas:
        per_parent[m["parent_id"]] = per_parent.get(m["parent_id"], 0) + 1
    with pg_store.connect() as conn:
        n_parents, parent_mean, parent_max = conn.execute(
            "SELECT count(*), avg(char_len), max(char_len) FROM parents WHERE parent_id = ANY(%s)",
            (list(per_parent),)).fetchone()
    return {
        **common,
        "strategy": "parent-child",
        "parent_unit": "## 제목(약관 조항, 용어집 용어)",
        "n_parents": n_parents,
        "chunks_per_parent_mean": round(len(docs) / len(per_parent), 2),
        "parents_split_2plus": sum(1 for v in per_parent.values() if v > 1),
        "parent_chars_mean": round(float(parent_mean), 1),
        "parent_chars_max": parent_max,
    }


# ---- 설정 ------------------------------------------------------------------

def retrieval_settings(method: str, args, k_max: int, strategy: str, hnsw: dict) -> dict:
    """검색 설정. 이 방식에 실제로 쓰이는 인자만 넣어 기준 대비 변경이 정확히 잡히게 한다."""
    s = {
        "method": method,
        "k": args.k,                 # 컨텍스트에 넣는 결과 수(대표 지표 기준)
        "depth": k_max,              # 지표를 계산하려고 받아 두는 결과 수(ks의 최댓값)
        "fetch_k": args.fetch_k,
        "hnsw_ef_search": hnsw.get("ef_search"),
        "result_unit": ("parent (같은 parent의 child는 가장 높은 순위로 묶음)" if strategy == "parent-child"
                        else "chunk (parent 없음, 청크 하나가 컨텍스트 한 자리)"),
    }
    if method == "mmr":
        s |= {"lambda_mult": args.lambda_mult, "candidate_pool": "2 x fetch_k"}
    if method in ("bm25", "hybrid"):
        s["bm25"] = {"tokenizer": "kiwipiepy", "pos_tags": sorted(BM25_TAGS), "algorithm": "BM25Okapi"}
    if method == "hybrid":
        s |= {"weight_vector": args.weight_vector, "fusion": "RRF", "rrf_k": RRF_K, "inputs": ["similarity", "bm25"]}
    return s


def data_files(files: list[str], doc_types: list[str]) -> list[Path]:
    """재현 확인용으로 해시를 남길 파일: 쓴 답안 파일 + 질문 유형 라벨 + 컬렉션 문서의 md."""
    return ([ANSWERS / f for f in files] + [LABELS]
            + [p for t in doc_types for p in (MD_DIR / t).glob("*.md")])


# ---- 실행 ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run", help="실험 이름(실험 ID에 들어감)")
    ap.add_argument("--note", default="", help="의도·가설(바뀐 설정은 --baseline으로 자동 기록)")
    ap.add_argument("--baseline", default="", help="기준 실험 ID(results/runs/ 아래 폴더 이름)")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--k", type=int, default=5, help="컨텍스트에 넣는 결과 수(기준 5). 대표 지표는 이 k로 본다")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10], help="지표를 함께 계산할 k들")
    ap.add_argument("--fetch-k", type=int, default=40)
    ap.add_argument("--lambda-mult", type=float, default=0.5)
    ap.add_argument("--weight-vector", type=float, default=0.5)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = ap.parse_args()

    ks = sorted(set(args.ks) | {args.k})
    k_max = ks[-1]
    retriever = Retriever(args.collection)
    chunking = chunking_info(retriever)
    cases, files = load_cases(set(chunking["source_doc_ids"]))
    doc_types, source_doc_ids = chunking.pop("doc_types"), chunking.pop("source_doc_ids")
    env = env_info(data_files(files, doc_types)) | {"collection_count": chunking["n_chunks"]}
    sweep_id = new_sweep_id(args.name)
    metric_keys = [f"{name}@{k}" for name in ("hit", "recall", "all_hit", "precision", "source_coverage",
                                              "key_phrase_recall", "dup_rate", "context_chars") for k in ks] + ["mrr"]

    summaries = {}
    for method in args.methods:
        exp_id = experiment_id(sweep_id, "retrieval", method)
        config = {
            "experiment_id": exp_id,
            "sweep_id": sweep_id,
            "stage": "retrieval",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "command": command_line(),
            "note": args.note,
            "settings": {
                "data": {
                    "answer_files": files,
                    "case_filter": "expected_source_ids에 컬렉션이 색인한 문서(source_doc_ids)의 근거가 있는 문항",
                    "n_cases": len(cases),
                    "n_cases_with_excluded_ids": sum(1 for c in cases if c["excluded_source_ids"]),
                    "doc_types": doc_types,
                    "source_doc_ids": source_doc_ids,
                },
                "chunking": chunking,
                "embedding": {"model": EMBEDDING_MODEL},
                "retrieval": retrieval_settings(method, args, k_max, chunking["strategy"],
                                                retriever.vs._collection.configuration_json.get("hnsw", {})),
                "eval": {
                    "ks": ks,
                    "relevance_unit": "결과의 source_id(조항·용어·PDF 쪽), 상위 k개 = 컨텍스트",
                    "labels": LABELS.relative_to(ROOT).as_posix(),
                    "source_coverage": "정답 근거 원문 중 컨텍스트에 들어간 글자 비율(줄 단위 매칭, 5자 미만 줄 제외)",
                    "key_phrase_recall": "must_include 중 정답 근거 원문에 그대로 있는 구절의 컨텍스트 포함률(없으면 제외)",
                    "dup_rate": "상위 k개 중 같은 source_id가 겹친 비율",
                },
            },
            "env": env,
        }
        run_dir = start_run(config, args.baseline)

        retriever.search(cases[0]["question"], method, k=k_max, fetch_k=args.fetch_k)  # 준비(BM25 색인 등) 시간 제외
        rows: list[dict] = []
        with (run_dir / "per_case.jsonl").open("w", encoding="utf-8") as fp:
            for c in cases:
                t = time.perf_counter()
                results = retriever.search(c["question"], method, k=k_max, fetch_k=args.fetch_k,
                                           lambda_mult=args.lambda_mult, weight_vector=args.weight_vector)
                latency = (time.perf_counter() - t) * 1000
                row = {
                    "experiment_id": exp_id, "sweep_id": sweep_id, **c,
                    "metrics": case_metrics(results, c, ks),
                    "latency_ms": round(latency, 1),
                    "retrieved": [{
                        "rank": r.rank, "source_id": r.source_id, "parent_id": r.parent_id,
                        "chunk_ids": r.matched_chunk_ids, "score": round(r.score, 6),
                        "source_file": r.source_file, "heading": r.heading,
                        "page_start": r.page_start, "page_end": r.page_end, "matched_pages": r.matched_pages,
                        "relevant": r.source_id in c["expected_source_ids"],
                        "content": r.content,
                    } for r in results],
                    "error": None,
                }
                rows.append(row)
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        groups, flat = {}, []
        for gname, key in GROUPS.items():
            for gval in sorted({key(r) for r in rows}, key=str):
                stats = aggregate([r for r in rows if key(r) == gval], metric_keys)
                groups.setdefault(gname, {})[gval] = stats
                flat.append({"experiment_id": exp_id, "group": gname, "value": gval, **stats})
        write_json(run_dir / "summary.json", {"experiment_id": exp_id, "groups": groups})
        write_csv(run_dir / "summary.csv", flat)
        summaries[method] = groups

        r = config["settings"]["retrieval"]
        overall = groups["all"]["all"]
        append_index({
            "experiment_id": exp_id, "sweep_id": sweep_id, "stage": "retrieval",
            "created_at": config["created_at"], "baseline_id": config["baseline_id"] or "",
            "changes": format_changes(config["changes"]), "env_changes": len(config["env_changes"]),
            "note": args.note,
            "collection": args.collection, "strategy": chunking["strategy"],
            "chunk_size": chunking["chunk_size"], "chunk_overlap": chunking["chunk_overlap"],
            "embedding_model": EMBEDDING_MODEL, "retrieval_method": method, "k": args.k, "fetch_k": args.fetch_k,
            "weight_vector": r.get("weight_vector", ""), "lambda_mult": r.get("lambda_mult", ""),
            "generation_model": "",
            "answer_files": " ".join(files), "n_cases": overall["n"],
            # 대표 지표: 컨텍스트 k 기준
            **{f"ctx_{name}": overall[f"{name}@{args.k}"] for name in
               ("hit", "recall", "precision", "source_coverage", "key_phrase_recall", "dup_rate", "context_chars")},
            "mrr": overall["mrr"],
            **{k: overall[k] for k in metric_keys if k != "mrr"},
            "latency_ms_mean": overall["latency_ms_mean"], "latency_ms_p50": overall["latency_ms_p50"],
            "latency_ms_p95": overall["latency_ms_p95"],
        })
        print(f"done {exp_id}" + (f"  changes: {format_changes(config['changes'])}" if config["changes"] else ""),
              flush=True)

    _print_table(summaries, args.methods, ks)
    print(f"\nsweep {sweep_id} -> {RUNS}")


def _print_table(summaries: dict, methods: list[str], ks: list[int]) -> None:
    cols = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks if k > 1] + ["mrr", "latency_ms_p50"]
    print("\n" + "retriever".ljust(12) + "".join(c.rjust(16) for c in cols))
    for m in methods:
        s = summaries[m]["all"]["all"]
        print(m.ljust(12) + "".join(f"{s[c]:16.3f}" if "latency" not in c else f"{s[c]:16.0f}" for c in cols))
    print("\nby difficulty (recall@5 / mrr)")
    for m in methods:
        d = summaries[m]["difficulty"]
        print("  " + m.ljust(12) + "  ".join(f"{g}={v.get('recall@5', float('nan')):.2f}/{v['mrr']:.2f} (n={v['n']})"
                                            for g, v in d.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()


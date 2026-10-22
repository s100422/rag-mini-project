"""대시보드 공통 데이터 로더. 평가 결과 폴더(results/, results_emb2/)를 읽기만 한다.

폴더 구조와 필드는 docs/experiment_records.md. 설정 이름 규칙(B0-base-rep2 등)과 라벨은 rag/report.py를 그대로 쓴다.
모든 함수는 결과 폴더 이름(SOURCES의 키)을 받고 st.cache_data로 캐시한다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent  # rag_mini_project/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.report import LABEL, NAME_RE, ORDER  # noqa: E402

SOURCES = {"emb-001 (gemini-embedding-001)": ROOT / "results", "emb-2 (gemini-embedding-2)": ROOT / "results_emb2"}
SET_NAME = {"base": "기본 세트 (용어집+약관, 100문항)", "ext": "확장 세트 (가이드북+추가 문서, 80문항)"}
BEHAVIOR_KO = {"answer": "답변", "correct": "정정", "ask": "되묻기", "abstain": "보류", "refuse": "거절"}
GROUP_KO = {"difficulty": "난이도", "set": "문항 종류", "category": "카테고리", "n_sources": "근거 수", "source_doc": "근거 문서",
            "question_type": "질문 유형", "needs_table": "표 질문 여부", "expected_group": "기대 행동"}
METRIC_KO = {"pass": "pass", "must_include_rate": "핵심 항목 포함", "coarse_match": "답변·보류 일치", "faithfulness": "근거 준수",
             "citation_precision": "인용 정밀도", "citation_recall": "인용 재현율"}
# 같은 설정을 반복해 돌렸을 때 '차이 없음'으로 보는 폭(%p, docs/experiment_plan.md 4.3절)
NOISE_PP = {"pass": 5.0, "must_include_rate": 3.0}
RET_RE = re.compile(r"^\d{8}_\d{6}_retrieval_((?:B0|E\d)(?:-[a-c])?)-(base|extended)-(similarity|bm25|hybrid|mmr)$")
RET_ORDER = ["B0", "E1-a", "E1-b", "E2-a", "E2-b", "E3-a", "E3-b", "E4-a", "E4-b", "E5-a", "E5-b", "E6-c"]
RET_LABEL = {"B0": "B0 Recursive 500/50", "E1-a": "E1-a 크기 300", "E1-b": "E1-b 크기 800", "E2-a": "E2-a 겹침 0",
             "E2-b": "E2-b 겹침 100", "E3-a": "E3-a 구조 단위", "E3-b": "E3-b 구조 + parent-child", "E4-a": "E4-a 제목 접두어",
             "E4-b": "E4-b 표 보존", "E5-a": "E5-a BM25", "E5-b": "E5-b hybrid", "E6-c": "E6-c MMR"}


def _runs_dir(source: str) -> Path:
    return SOURCES[source] / "runs"


def available_sources() -> list[str]:
    """실행 기록이 있는 결과 폴더만."""
    return [k for k, p in SOURCES.items() if (p / "runs").is_dir() and any((p / "runs").iterdir())]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def answer_runs(source: str) -> pd.DataFrame:
    """답변 단계(rag·agent) 실행 한 줄씩. 전체 지표는 summary.json의 all/all.

    열: run_id, setting(B0·E3-b·A2 …), label, set_name(base|ext), rep(이름 끝 반복 표시), stage, k,
        n, n_errors, n_blocked, pass, must_include_rate, coarse_match, faithfulness, citation_*, hit@5, source_coverage@5,
        input_tokens_mean, output_tokens_mean, thinking_tokens_mean, llm_calls_mean, total_ms_p50, total_ms_p95
    """
    rows = []
    for d in sorted(_runs_dir(source).glob("*")):
        m = NAME_RE.match(d.name)
        if not m or not (d / "summary.json").exists():
            continue
        name = m.group(2)
        rep = re.search(r"-(r2|rep\d+)$", name)
        setting, _, which = re.sub(r"-(r2|rep\d+)$", "", name).rpartition("-")
        if which not in ("base", "ext") or setting not in LABEL:
            continue
        cfg = _read(d / "config.json")["settings"]
        a = _read(d / "summary.json")["groups"]["all"]["all"]
        rows.append({"run_id": d.name, "setting": setting, "label": LABEL[setting], "set_name": which,
                     "rep": rep.group(1) if rep else "r1", "stage": m.group(1), "k": cfg["retrieval"]["k"], **a})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def answer_groups(source: str) -> pd.DataFrame:
    """답변 단계 그룹별 지표(긴 형식). 열: run_id, setting, set_name, group_type, group, n + 지표들."""
    rows = []
    runs = answer_runs(source)
    for r in runs.itertuples():
        for gtype, groups in _read(_runs_dir(source) / r.run_id / "summary.json")["groups"].items():
            for g, vals in groups.items():
                rows.append({"run_id": r.run_id, "setting": r.setting, "set_name": r.set_name, "group_type": gtype,
                             "group": g, **vals})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def retrieval_runs(source: str) -> pd.DataFrame:
    """검색 단계 실행 한 줄씩(같은 실험 이름이면 가장 나중 것). 열: key(B0·E1-a …), label, set_name, method, run_id, k별 지표들."""
    best: dict[tuple, dict] = {}
    for d in sorted(_runs_dir(source).glob("*")):
        m = RET_RE.match(d.name)
        if not m or not (d / "summary.json").exists():
            continue
        key = m.group(1)
        a = _read(d / "summary.json")["groups"]["all"]["all"]
        best[(key, m.group(2))] = {"run_id": d.name, "key": key, "label": RET_LABEL.get(key, key),
                                   "set_name": "base" if m.group(2) == "base" else "ext", "method": m.group(3), **a}
    return pd.DataFrame(list(best.values()))


def settings_table(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """설정별 반복 평균·최소·최대(지표는 0~1 값). ORDER 순서. 열: setting, label, runs, mean, min, max."""
    if runs.empty:
        return pd.DataFrame(columns=["setting", "label", "runs", "mean", "min", "max"])
    g = runs.groupby("setting")[metric].agg(["mean", "min", "max", "count"]).reset_index()
    g["label"] = g["setting"].map(LABEL)
    g["order"] = g["setting"].map({s: i for i, s in enumerate(ORDER)})
    return g.sort_values("order").rename(columns={"count": "runs"})[["setting", "label", "runs", "mean", "min", "max"]]


@st.cache_data(show_spinner=False)
def load_cases(source: str, run_id: str) -> dict[str, dict]:
    """문항별 결과 {case_id: per_case 줄}. 용량이 커서 문항 탐색 탭에서만 부른다."""
    path = _runs_dir(source) / run_id / "per_case.jsonl"
    return {r["case_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


@st.cache_data(show_spinner=False)
def reference_answers() -> dict[str, str]:
    """문항별 기준 답안(03_answers). 문항 탐색 탭에서 쓴다."""
    out = {}
    for f in sorted((ROOT / "03_answers").glob("*cases.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["case_id"]] = r["reference_answer"]
    return out

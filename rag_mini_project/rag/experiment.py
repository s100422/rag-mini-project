"""실험 기록 공통 코드(설계: docs/experiment_records.md).

results/runs/<experiment_id>/ 에 config.json·per_case.jsonl·summary.json/csv·prompts/를 쓰고,
results/experiments.csv에 실험당 한 줄을 쌓는다. 검색 평가·기본 RAG·Agent가 같이 쓴다.
코드 버전은 git 커밋 해시로 남긴다(커밋하지 않은 변경이 있으면 -dirty).
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from .config import ROOT

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
INDEX = RESULTS / "experiments.csv"
PACKAGES = ["chromadb", "langchain-chroma", "langchain-core", "langchain-google-genai", "google-genai",
            "langchain-text-splitters", "kiwipiepy", "rank-bm25", "psycopg", "pymupdf", "langsmith"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_sweep_id(name: str, now: datetime | None = None) -> str:
    return f"{(now or datetime.now()):%Y%m%d_%H%M%S}_{name}"


def experiment_id(sweep_id: str, stage: str, variant: str) -> str:
    """sweep ID(날짜_시각_이름)에 단계·변형을 끼운다: 20261021_103000_retrieval_base-hybrid"""
    date, time_, name = sweep_id.split("_", 2)
    return f"{date}_{time_}_{stage}_{name}-{variant}"


# ---- 재현 정보 --------------------------------------------------------------

def code_version() -> str:
    """git 커밋 해시. 커밋하지 않은 변경이 있으면 뒤에 -dirty를 붙인다(git 없으면 unknown)."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def env_info(data_files: list[Path]) -> dict:
    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "code_version": code_version(),
        "data_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(data_files))},
    }


def save_prompt(run_dir: Path, name: str, text: str) -> dict:
    """프롬프트 원문을 실험 폴더에 남기고 config에 넣을 참조를 돌려준다."""
    path = run_dir / "prompts" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"file": path.relative_to(run_dir).as_posix(), "sha256": sha256_text(text)}


# ---- 기준 대비 변경 ---------------------------------------------------------

def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            out |= _flatten(v, key)
        else:
            out[key] = v
    return out


def diff_settings(baseline: dict, current: dict) -> list[dict]:
    """두 config의 settings에서 달라진 항목(점 경로)만 뽑는다."""
    a, b = _flatten(baseline.get("settings", {})), _flatten(current.get("settings", {}))
    return [{"path": k, "baseline": a.get(k), "value": b.get(k)}
            for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]


def diff_env(baseline: dict, current: dict) -> list[str]:
    """코드 버전과 데이터 파일 중 기준 실험과 다른 것(재현성 확인용)."""
    a_env, b_env = baseline.get("env", {}), current.get("env", {})
    out = []
    if a_env.get("code_version") != b_env.get("code_version"):
        out.append(f"code_version:{a_env.get('code_version')}→{b_env.get('code_version')}")
    a, b = a_env.get("data_files", {}), b_env.get("data_files", {})
    out += [f"data_files:{k}" for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]
    return out


def load_config(exp_id: str) -> dict:
    path = RUNS / exp_id / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"기준 실험이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_changes(changes: list[dict]) -> str:
    return "; ".join(f"{c['path']}: {c['baseline']}→{c['value']}" for c in changes)


# ---- 쓰기 -------------------------------------------------------------------

def start_run(config: dict, baseline_id: str | None) -> Path:
    """실험 폴더를 만들고 config(기준 대비 변경 포함)를 쓴다."""
    run_dir = RUNS / config["experiment_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    config["baseline_id"] = baseline_id or None
    if baseline_id:
        base = load_config(baseline_id)
        config["changes"] = diff_settings(base, config)
        config["env_changes"] = diff_env(base, config)
    else:
        config["changes"], config["env_changes"] = [], []
    write_json(run_dir / "config.json", config)
    return run_dir


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_index(row: dict) -> None:
    """experiments.csv에 한 줄 추가(열이 늘면 기존 줄은 빈 칸)."""
    old = []
    if INDEX.exists():
        with INDEX.open(newline="", encoding="utf-8-sig") as fp:
            old = list(csv.DictReader(fp))
    write_csv(INDEX, old + [row])


def command_line() -> str:
    return " ".join(["python -m rag." + Path(sys.argv[0]).stem, *sys.argv[1:]])

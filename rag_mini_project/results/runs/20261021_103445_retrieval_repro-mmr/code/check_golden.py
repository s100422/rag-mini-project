"""골든셋(03_answers/*.jsonl)의 expected_source_ids가 저장된 parent·child와 연결되는지 확인한다.

사용법: python -m rag.check_golden [--collection base_c350_o50]
"""
import argparse
import json
import sys
from collections import Counter

from .config import ROOT
from . import pg_store

ANSWERS = ROOT / "03_answers"
INDEXED = {"kofia-terms": "약관", "fsc-glossary": "용어집"}   # parent-child로 색인한 문서의 근거 ID 앞부분
PAGE_LEVEL = "-pdf-"   # 쪽 단위 근거 ID(가이드북·추가 문서): parent 없이 Chroma 청크에만 있다


def load_cases() -> list[dict]:
    cases = []
    for f in sorted(ANSWERS.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line) | {"_file": f.name})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="base_c350_o50")
    args = ap.parse_args()

    cases = load_cases()
    with pg_store.connect() as conn:
        pg_ids = {r[0] for r in conn.execute("SELECT source_id FROM parents WHERE source_id IS NOT NULL")}

    from .chroma_store import get_vectorstore
    meta = get_vectorstore(args.collection).get(include=["metadatas"])["metadatas"]
    chroma_ids = {m.get("source_id") for m in meta}

    kinds = Counter()
    missing = []
    for c in cases:
        for sid in c["expected_source_ids"]:
            if PAGE_LEVEL in sid:
                kinds[sid.split(PAGE_LEVEL)[0]] += 1
                if sid not in chroma_ids:
                    missing.append((c["case_id"], sid, None, False))
                continue
            prefix = next((x for x in INDEXED if sid.startswith(x)), None)
            kinds[prefix or sid] += 1
            if sid not in pg_ids or sid not in chroma_ids:
                missing.append((c["case_id"], sid, sid in pg_ids, sid in chroma_ids))

    print("근거 ID 종류:", dict(kinds))
    for prefix, n in kinds.items():
        print(f"{prefix} 근거 ID {n}개 중 연결 안 됨 {sum(m[1].startswith(prefix) for m in missing)}개")
    for case_id, sid, in_pg, in_chroma in missing[:30]:
        print(f"  {case_id} {sid} postgres={in_pg} chroma={in_chroma}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

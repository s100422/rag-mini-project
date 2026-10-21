"""md 로드 → (parent를 Postgres에 저장) → 청킹 → Chroma 저장.

--strategy auto(기본): 예전 방식. 약관·용어집은 조항·용어 parent-child, 가이드북·추가 문서는 쪽 단위 청킹.
--strategy recursive|structure|parent-child: 실험용 전략(rag/doc_chunking.py, docs/experiment_plan.md).
전략과 설정은 컬렉션 메타데이터에 남아 평가(rag.evaluate)가 설정으로 읽는다.

사용법:
    python -m rag.build_index --doc-type standard_terms glossary --chunk-size 350 --collection base_c350_o50
    python -m rag.build_index --doc-type life_guide additional_p4l --collection extended_c500_o50
    python -m rag.build_index --doc-type standard_terms glossary --strategy recursive --collection base_rec_c500_o50
    python -m rag.build_index --doc-type life_guide additional_p4l --strategy recursive --table-aware \\
        --collection extended_rec_table_c500_o50
    python -m rag.build_index ... --skip-chroma        # 임베딩 없이 청킹 통계만
"""
import argparse
import statistics
import sys

from .chunking import SEPARATORS, ChunkConfig, split_pages, split_parents
from .config import CHROMA_DIR, MD_DIR, pg_settings
from .doc_chunking import STRATEGIES, chunk_docs, load_docs, unit_parents
from .md_parser import load_pages, load_parents
from . import pg_store

PAGE_DOC_TYPES = {"life_guide", "additional_p4l"}
UNITS = "약관=조항, 용어집=용어, 가이드북=제목 구간, 추가 문서=쪽"


def save_parents(parents) -> None:
    cfg = pg_settings()
    pg_store.ensure_database(cfg)
    with pg_store.connect(cfg) as conn:
        pg_store.ensure_schema(conn)
        n = pg_store.replace_parents(conn, parents)
    print(f"saved {n} parents -> postgres {cfg.dbname}.parents")


def build_auto(doc_types: list[str], chunk_cfg: ChunkConfig, skip_db: bool) -> tuple[list, dict]:
    parent_types = [t for t in doc_types if t not in PAGE_DOC_TYPES]
    page_types = [t for t in doc_types if t in PAGE_DOC_TYPES]
    docs = []
    if parent_types:
        parents = [p for t in parent_types for p in load_parents(MD_DIR / t, t)]
        print(f"parents: {len(parents)}")
        if not skip_db:
            save_parents(parents)
        docs += split_parents(parents, chunk_cfg)
    if page_types:
        docs += split_pages([p for t in page_types for p in load_pages(MD_DIR / t, t)], chunk_cfg)
    strategy = "parent-child" if parent_types and not page_types else "page" if page_types and not parent_types else "auto"
    unit = "조항·용어 parent" if strategy == "parent-child" else "PDF 쪽" if strategy == "page" else ""
    return docs, {"chunk_strategy": strategy, "chunk_unit": unit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-type", nargs="+", default=["standard_terms"], help="md/ 아래 폴더 이름(여러 개 가능)")
    ap.add_argument("--strategy", default="auto", choices=["auto", *STRATEGIES])
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--title-prefix", action="store_true", help="청크 앞에 [문서 제목 > 상위 제목] (E4-a)")
    ap.add_argument("--table-aware", action="store_true", help="표는 행 경계에서 자르고 제목·머리행·각주 유지 (E4-b)")
    ap.add_argument("--collection", help="Chroma 컬렉션 이름 (기본: <doc-type들을 _로 이음>_c<size>_o<overlap>)")
    ap.add_argument("--skip-chroma", action="store_true")
    args = ap.parse_args()

    chunk_cfg = ChunkConfig(args.chunk_size, args.chunk_overlap)
    if args.strategy == "auto":
        if args.title_prefix or args.table_aware:
            ap.error("--title-prefix, --table-aware는 recursive·structure 전략에서만 쓴다")
        docs, meta = build_auto(args.doc_type, chunk_cfg, args.skip_chroma)
    else:
        texts = [d for t in args.doc_type for d in load_docs(t)]
        print(f"documents: {len(texts)}, chars {sum(len(d.text) for d in texts):,}")
        if args.strategy == "parent-child":
            if args.title_prefix or args.table_aware:
                ap.error("--title-prefix, --table-aware는 recursive·structure 전략에서만 쓴다")
            parents = unit_parents(texts)
            print(f"parents: {len(parents)}")
            if not args.skip_chroma:
                save_parents(parents)
            docs = split_parents(parents, chunk_cfg)
        else:
            docs = chunk_docs(texts, args.strategy, chunk_cfg, args.title_prefix, args.table_aware)
        meta = {"chunk_strategy": args.strategy,
                "chunk_unit": "문서 전체" if args.strategy == "recursive" else UNITS}
    meta |= {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
             "title_prefix": args.title_prefix, "table_aware": args.table_aware,
             "separators": " | ".join(repr(s) for s in SEPARATORS), "doc_types": ",".join(args.doc_type)}

    lens = [len(d.page_content) for d in docs]
    multi = sum(1 for d in docs if len(d.metadata.get("source_ids", [])) > 1)
    print(f"chunks ({chunk_cfg.name}): {len(docs)}, len median {statistics.median(lens):.0f}, max {max(lens)}, "
          f"chunks over 2+ sources: {multi}")

    if args.skip_chroma:
        return
    from .chroma_store import get_vectorstore, replace_chunks  # 임베딩 없이 돌릴 때는 불러오지 않는다

    collection = args.collection or f"{'_'.join(args.doc_type)}_{chunk_cfg.name}"
    n = replace_chunks(get_vectorstore(collection, meta), docs)
    print(f"saved {n} chunks -> chroma {CHROMA_DIR} [{collection}]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

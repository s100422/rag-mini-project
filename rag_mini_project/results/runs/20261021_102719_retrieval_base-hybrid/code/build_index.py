"""md 로드 → parent(Postgres) 저장 → child 청킹 → Chroma 저장.
PAGE_DOC_TYPES(가이드북·추가 문서)는 parent 없이 쪽 단위로 바로 청킹해 Chroma에만 저장한다.

사용법:
    python -m rag.build_index                          # standard_terms, 500자/겹침 50
    python -m rag.build_index --chunk-size 300 --chunk-overlap 30
    python -m rag.build_index --skip-chroma            # 임베딩 없이 parent 저장·청킹만
    python -m rag.build_index --doc-type standard_terms glossary --chunk-size 350 --collection base_c350_o50
    python -m rag.build_index --doc-type life_guide additional_p4l --collection extended_c500_o50
"""
import argparse
import statistics
import sys

from .chunking import ChunkConfig, split_pages, split_parents
from .config import CHROMA_DIR, MD_DIR, pg_settings
from .md_parser import load_pages, load_parents
from . import pg_store

PAGE_DOC_TYPES = {"life_guide", "additional_p4l"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-type", nargs="+", default=["standard_terms"], help="md/ 아래 폴더 이름(여러 개 가능)")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--collection", help="Chroma 컬렉션 이름 (기본: <doc-type들을 _로 이음>_c<size>_o<overlap>)")
    ap.add_argument("--skip-chroma", action="store_true")
    args = ap.parse_args()

    chunk_cfg = ChunkConfig(args.chunk_size, args.chunk_overlap)
    parent_types = [t for t in args.doc_type if t not in PAGE_DOC_TYPES]
    page_types = [t for t in args.doc_type if t in PAGE_DOC_TYPES]
    docs = []

    if parent_types:
        parents = [p for t in parent_types for p in load_parents(MD_DIR / t, t)]
        for t in parent_types:
            ps = [p for p in parents if p.doc_type == t]
            print(f"parents [{t}]: {len(ps)} (docs {len({p.doc_id for p in ps})})")

        cfg = pg_settings()
        pg_store.ensure_database(cfg)
        with pg_store.connect(cfg) as conn:
            pg_store.ensure_schema(conn)
            n = pg_store.replace_parents(conn, parents)
        print(f"saved {n} parents -> postgres {cfg.dbname}.parents")

        child = split_parents(parents, chunk_cfg)
        multi = sum(1 for p in parents if sum(d.metadata["parent_id"] == p.parent_id for d in child) > 1)
        print(f"parents split into 2+ chunks: {multi}")
        docs += child

    if page_types:
        pages = [p for t in page_types for p in load_pages(MD_DIR / t, t)]
        for t in page_types:
            ps = [p for p in pages if p.doc_type == t]
            print(f"pages [{t}]: {len(ps)} (docs {len({p.doc_id for p in ps})})")
        docs += split_pages(pages, chunk_cfg)

    lens = [len(d.page_content) for d in docs]
    print(f"chunks ({chunk_cfg.name}): {len(docs)}, len median {statistics.median(lens):.0f}, max {max(lens)}")

    if args.skip_chroma:
        return
    from .chroma_store import get_vectorstore, replace_chunks  # 임베딩 없이 돌릴 때는 불러오지 않는다

    collection = args.collection or f"{'_'.join(args.doc_type)}_{chunk_cfg.name}"
    n = replace_chunks(get_vectorstore(collection), docs)
    print(f"saved {n} chunks -> chroma {CHROMA_DIR} [{collection}]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

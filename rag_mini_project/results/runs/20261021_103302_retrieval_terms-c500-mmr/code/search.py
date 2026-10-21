"""검색기를 터미널에서 시험한다.

사용법:
    python -m rag.search "신용거래보증금은 어떻게 납부하나요?" --method hybrid -k 5
    python -m rag.search "…" --method all          # 네 방식 비교
    python -m rag.search "…" --json                # 결과 전체(parent 원문 포함)를 JSON으로
"""
import argparse
import json
import sys

from .retrievers import DEFAULT_COLLECTION, Retriever

METHODS = ["similarity", "mmr", "bm25", "hybrid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--method", default="hybrid", choices=METHODS + ["all"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--fetch-k", type=int, default=40)
    ap.add_argument("--lambda-mult", type=float, default=0.5, help="MMR: 1이면 유사도만, 0이면 다양성만")
    ap.add_argument("--weight-vector", type=float, default=0.5, help="hybrid: 벡터 쪽 RRF 가중치")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = Retriever(args.collection)
    for method in METHODS if args.method == "all" else [args.method]:
        results = r.search(args.query, method, k=args.k, fetch_k=args.fetch_k,
                           lambda_mult=args.lambda_mult, weight_vector=args.weight_vector)
        if args.json:
            print(json.dumps([x.to_dict() for x in results], ensure_ascii=False, indent=2))
            continue
        print(f"[{method}]")
        for x in results:
            print(f"  {x.rank}. {x.score:.4f} {x.source_id} | {x.source_file} p.{x.page_start}-{x.page_end} | {x.heading}"
                  f" | chunks {len(x.matched_chunk_ids)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

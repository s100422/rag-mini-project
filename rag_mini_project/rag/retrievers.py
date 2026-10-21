"""child 청크를 검색하고, 결과를 parent 단위로 묶어 메타데이터와 parent 원문을 돌려준다.

검색 방식
- similarity : Chroma 코사인 유사도
- mmr        : Chroma MMR (유사도와 다양성 절충, lambda_mult)
- bm25       : Kiwi 형태소 토큰 + BM25Okapi (child 전체를 메모리에 색인)
- hybrid     : similarity와 bm25 순위를 RRF로 합침 (가중치 조절 가능)

결과 단위
- child를 순위대로 받아 같은 parent는 처음 나온(가장 높은) 순위로 하나만 남긴다.
- parent가 없는 청크(가이드북·추가 문서)는 청크 자체가 결과 하나다(parent_id 자리에 chunk_id).
- 상위 k개 parent를 돌려준다. 맞은 child ID·쪽은 matched_* 에 남겨 평가에 쓴다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from . import pg_store
from .chroma_store import get_vectorstore

DEFAULT_COLLECTION = "standard_terms_c500_o50"
RRF_K = 60

# BM25에 쓸 내용어 품사: 명사, 수사, 어근, 동사·형용사 어간, 외국어·숫자·한자
BM25_TAGS = {"NNG", "NNP", "NR", "XR", "VV", "VA", "SL", "SN", "SH"}


@dataclass
class ChildHit:
    chunk_id: str
    parent_id: str
    score: float          # 방식마다 의미가 다름(similarity: 코사인 유사도, bm25: BM25 점수, hybrid: RRF 점수, mmr: 순위 역수)
    metadata: dict
    text: str = ""        # child 본문(parent가 없을 때 결과 본문으로 쓴다)


def _hit(doc_text: str, meta: dict, score: float) -> ChildHit:
    return ChildHit(meta["chunk_id"], meta.get("parent_id", meta["chunk_id"]), score, meta, doc_text)


@dataclass
class RetrievalResult:
    rank: int
    retriever: str
    score: float                    # 이 parent의 가장 높은 child 점수
    parent_id: str
    source_id: str | None           # 골든셋 expected_source_ids와 비교
    source_doc_id: str | None
    source_file: str
    doc_title: str
    heading: str
    page_start: int                 # parent 쪽 범위
    page_end: int
    source_ids: list[str] = field(default_factory=list)          # 이 결과가 걸친 근거 ID 전부(문서 전체를 자른 청크는 여러 개)
    matched_chunk_ids: list[str] = field(default_factory=list)  # 이 parent에서 걸린 child (순위순)
    matched_pages: list[int] = field(default_factory=list)      # 걸린 child가 있는 쪽
    content: str = ""               # parent 원문(답변 생성 컨텍스트)

    def to_dict(self) -> dict:
        return asdict(self)


class KiwiTokenizer:
    def __init__(self) -> None:
        self.kiwi = Kiwi()

    def __call__(self, text: str) -> list[str]:
        return [t.form.lower() for t in self.kiwi.tokenize(text) if t.tag.split("-")[0] in BM25_TAGS]


class Retriever:
    def __init__(self, collection: str = DEFAULT_COLLECTION):
        self.collection = collection
        self.vs = get_vectorstore(collection)
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[tuple[str, dict]] = []
        self._tokenizer: KiwiTokenizer | None = None

    # ---- child 검색 -------------------------------------------------------

    def similarity(self, query: str, n: int) -> list[ChildHit]:
        docs = self.vs.similarity_search_with_score(query, k=n)
        # 코사인 거리(0~2) → 유사도
        return [_hit(d.page_content, d.metadata, 1 - dist) for d, dist in docs]

    def mmr(self, query: str, n: int, fetch_k: int = 40, lambda_mult: float = 0.5) -> list[ChildHit]:
        docs = self.vs.max_marginal_relevance_search(query, k=n, fetch_k=max(fetch_k, n), lambda_mult=lambda_mult)
        return [_hit(d.page_content, d.metadata, 1 / (i + 1)) for i, d in enumerate(docs)]

    def bm25(self, query: str, n: int) -> list[ChildHit]:
        self._ensure_bm25()
        scores = self._bm25.get_scores(self._tokenizer(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [_hit(self._bm25_docs[i][0], self._bm25_docs[i][1], float(scores[i]))
                for i in order if scores[i] > 0]

    def hybrid(self, query: str, n: int, fetch_k: int = 40, weight_vector: float = 0.5) -> list[ChildHit]:
        """RRF: score = w·1/(60+순위_벡터) + (1-w)·1/(60+순위_BM25)."""
        lists = [(self.similarity(query, fetch_k), weight_vector),
                 (self.bm25(query, fetch_k), 1 - weight_vector)]
        fused: dict[str, float] = {}
        meta: dict[str, ChildHit] = {}
        for hits, w in lists:
            for rank, h in enumerate(hits, start=1):
                fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + w / (RRF_K + rank)
                meta.setdefault(h.chunk_id, h)
        order = sorted(fused, key=fused.get, reverse=True)[:n]
        return [ChildHit(cid, meta[cid].parent_id, fused[cid], meta[cid].metadata, meta[cid].text) for cid in order]

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        data = self.vs.get(include=["documents", "metadatas"])
        self._tokenizer = KiwiTokenizer()
        self._bm25_docs = list(zip(data["documents"], data["metadatas"]))
        self._bm25 = BM25Okapi([self._tokenizer(doc) for doc in data["documents"]])

    # ---- parent 단위 결과 -------------------------------------------------

    def search(self, query: str, method: str = "similarity", k: int = 5,
               fetch_k: int = 40, lambda_mult: float = 0.5, weight_vector: float = 0.5) -> list[RetrievalResult]:
        """상위 k개 parent를 돌려준다. parent를 k개 채우도록 child는 fetch_k개까지 받는다."""
        n = max(fetch_k, k)
        if method == "similarity":
            hits = self.similarity(query, n)
        elif method == "mmr":
            hits = self.mmr(query, n, fetch_k=n * 2, lambda_mult=lambda_mult)
        elif method == "bm25":
            hits = self.bm25(query, n)
        elif method == "hybrid":
            hits = self.hybrid(query, n, fetch_k=n, weight_vector=weight_vector)
        else:
            raise ValueError(f"unknown method: {method}")
        return self._group_by_parent(hits, k, method)

    def _group_by_parent(self, hits: list[ChildHit], k: int, method: str) -> list[RetrievalResult]:
        grouped: dict[str, list[ChildHit]] = {}
        for h in hits:
            if h.parent_id in grouped or len(grouped) < k:
                grouped.setdefault(h.parent_id, []).append(h)
        with pg_store.connect() as conn:
            parents = pg_store.get_parents(conn, list(grouped))
        results = []
        for rank, (pid, hs) in enumerate(grouped.items(), start=1):
            if pid not in parents:  # parent 없는 청크: 청크 메타데이터·본문을 그대로 쓴다
                m = hs[0].metadata
                p = {"source_id": m.get("source_id"), "source_doc_id": m.get("source_doc_id"),
                     "source_ids": m.get("source_ids"),
                     "source_file": m["source_file"], "doc_title": m.get("doc_title", m["doc_id"]),
                     "heading": m.get("heading", ""), "page_start": m["page_start"],
                     "page_end": m["page_end"], "content": hs[0].text}
            else:
                p = parents[pid]
            pages = sorted({pg for h in hs for pg in range(h.metadata["page_start"], h.metadata["page_end"] + 1)})
            source_ids = p.get("source_ids") or ([p["source_id"]] if p["source_id"] else [])
            results.append(RetrievalResult(
                rank=rank, retriever=method, score=hs[0].score, parent_id=pid,
                source_id=p["source_id"], source_doc_id=p["source_doc_id"], source_ids=list(source_ids),
                source_file=p["source_file"], doc_title=p["doc_title"], heading=p["heading"],
                page_start=p["page_start"], page_end=p["page_end"],
                matched_chunk_ids=[h.chunk_id for h in hs], matched_pages=pages,
                content=p["content"],
            ))
        return results

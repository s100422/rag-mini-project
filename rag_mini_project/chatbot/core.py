"""챗봇 백엔드: 최종 설정으로 질문 하나에 답한다.

- 검색: 구조 단위 parent-child(E3-b) 컬렉션 두 개(용어집·약관 / 가이드북·추가 문서)를 모두 검색해 점수순으로 합친 상위 k개.
- 답변: Agent v2 + 시스템 프롬프트 v2(A4: 질문 분석 → 검색 → 답변, 재검색 없음). 구조는 docs/agent_graph.md.
평가에서는 두 문서 묶음을 따로 검색했다. 여기서는 사용자가 어느 문서에 있는지 모르고 물으므로 합쳐서 검색한다(합친 조건은 평가하지 않았다).
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # rag_mini_project/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import agent  # noqa: E402
from rag.rag import page_label  # noqa: E402
from rag.retrievers import Retriever  # noqa: E402

COLLECTIONS = ["base_pc_c500_o50", "extended_pc_c500_o50"]  # E3-b: 용어집+약관 / 가이드북+추가 문서
AGENT_OPTIONS = "retry=0,v2=1,sys2=1"                        # A4
SEARCH = dict(method="similarity", k=5, fetch_k=40, lambda_mult=0.5, weight_vector=0.5)
REQUEST_TYPE_KO = {"normal": "일반 질문", "missing_info": "정보 부족(되묻기)", "advice_or_misuse": "투자 권유·부적절한 요청(거절)"}


class CombinedRetriever:
    """여러 컬렉션을 같은 질의로 검색해 코사인 유사도 점수로 합친 상위 k개를 돌려준다(같은 임베딩이라 점수를 비교할 수 있다)."""

    def __init__(self, collections: list[str]):
        self.retrievers = [Retriever(c) for c in collections]

    def search(self, query: str, **search):
        pooled = [r for ret in self.retrievers for r in ret.search(query, **search)]
        pooled.sort(key=lambda r: r.score, reverse=True)
        top = pooled[:search.get("k", 5)]
        for i, r in enumerate(top, 1):
            r.rank = i
        return top


@dataclass
class Source:
    number: int
    file: str
    pages: str
    heading: str
    content: str
    cited: bool


@dataclass
class Reply:
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    request_type: str = ""
    seconds: float = 0.0
    llm_calls: int = 0
    error: str = ""


def clean_heading(heading: str) -> str:
    """쪽 단위로 자른 문서는 제목 자리에 내부 ID(문서-pdf-쪽)가 들어 있다. 사람이 읽을 수 없으니 뺀다."""
    return "" if re.fullmatch(r".+-pdf-\d+", heading or "") else heading


def load():
    """검색기와 Agent를 한 번 만든다(앱에서 캐시해 쓴다)."""
    return CombinedRetriever(COLLECTIONS), agent.make_runner(AGENT_OPTIONS)


def ask(question: str, retriever: CombinedRetriever, run) -> Reply:
    """질문 하나를 처리한다. API 오류는 예외 대신 reply.error로 돌려준다."""
    t = time.perf_counter()
    try:
        rec = run(question, retriever, SEARCH)
    except Exception as e:  # noqa: BLE001  화면에는 짧은 안내만 보인다
        return Reply(error=f"{type(e).__name__}: {str(e)[:200]}", seconds=time.perf_counter() - t)
    reply = Reply(seconds=time.perf_counter() - t, llm_calls=rec.usage.llm_calls)
    analyze = next((s for s in rec.steps if s["node"] == "analyze"), None)
    reply.request_type = REQUEST_TYPE_KO.get(analyze["request_type"], "") if analyze else ""
    if rec.blocked:
        reply.answer = "이 요청은 안전 정책에 따라 답변할 수 없습니다. 질문을 바꿔서 다시 시도해 주세요."
        return reply
    reply.answer = rec.answer
    cited = set(rec.citations)
    reply.sources = [Source(i, r.source_file, page_label(r), clean_heading(r.heading), r.content, i in cited)
                     for i, r in enumerate(rec.results, 1)]
    return reply

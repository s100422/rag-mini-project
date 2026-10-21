"""child 청크를 Gemini로 임베딩해 Chroma(chroma_db/)에 저장한다."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CHROMA_DIR, EMBEDDING_MODEL

BATCH = 100
# HNSW 탐색 폭. 기본 100이면 MMR처럼 후보를 80개 받을 때 끝부분이 실행마다 달라진다.
# 컬렉션이 천 개 남짓이라 1000이면 전수 검색과 같은 결과가 나온다(실험 재현성).
EF_SEARCH = 1000


def get_vectorstore(collection: str, metadata: dict | None = None) -> Chroma:
    """metadata(청킹 전략·설정)는 컬렉션을 새로 만들 때만 저장된다. 평가가 이 값을 설정으로 읽는다."""
    return Chroma(
        collection_name=collection,
        # 문서는 RETRIEVAL_DOCUMENT, 질의는 RETRIEVAL_QUERY로 임베딩된다(langchain-google-genai 기본값)
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, request_options={"timeout": 120}),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL, **(metadata or {})},
    )


def replace_chunks(vs: Chroma, docs: list[Document]) -> int:
    """docs에 들어 있는 문서(doc_id)의 기존 청크를 지우고 새로 넣는다."""
    doc_ids = sorted({d.metadata["doc_id"] for d in docs})
    vs._collection.modify(configuration={"hnsw": {"ef_search": EF_SEARCH}})
    vs.delete(where={"doc_id": {"$in": doc_ids}})
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        vs.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        print(f"  embedded {min(i + BATCH, len(docs))}/{len(docs)}", flush=True)
    return len(docs)

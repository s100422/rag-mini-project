"""child 청크를 Gemini로 임베딩해 Chroma(chroma_db/)에 저장한다."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CHROMA_DIR, EMBEDDING_MODEL

BATCH = 100


def get_vectorstore(collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        # 문서는 RETRIEVAL_DOCUMENT, 질의는 RETRIEVAL_QUERY로 임베딩된다(langchain-google-genai 기본값)
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


def replace_chunks(vs: Chroma, docs: list[Document]) -> int:
    """docs에 들어 있는 문서(doc_id)의 기존 청크를 지우고 새로 넣는다."""
    doc_ids = sorted({d.metadata["doc_id"] for d in docs})
    vs.delete(where={"doc_id": {"$in": doc_ids}})
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        vs.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        print(f"  embedded {min(i + BATCH, len(docs))}/{len(docs)}", flush=True)
    return len(docs)

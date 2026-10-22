"""RAG 평가 결과 대시보드. 실행: rag_mini_project/ 에서  streamlit run dashboard/app.py

읽기 전용. 결과 폴더(results/, results_emb2/)의 summary.json·per_case.jsonl을 읽는다(dashboard/data.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from dashboard.data import SET_NAME, available_sources  # noqa: E402
from dashboard.tabs import answers, cases, cost, embeddings, noise, overview, retrieval  # noqa: E402

st.set_page_config(page_title="RAG 평가 결과", layout="wide")
st.title("RAG 평가 결과")

sources = available_sources()
if not sources:
    st.error("결과 폴더(results/, results_emb2/)에 실행 기록이 없습니다.")
    st.stop()
with st.sidebar:
    st.header("보기 설정")
    source = st.selectbox("임베딩(결과 폴더)", sources)
    which = st.radio("질문 세트", list(SET_NAME), format_func=SET_NAME.get)
    st.caption("세트끼리는 지표를 비교하지 않는다(검색 범위가 다르다).")

TABS = [("개요", overview), ("검색 단계", retrieval), ("답변 비교", answers), ("비용 대비 품질", cost),
        ("반복 오차", noise), ("문항 탐색", cases), ("임베딩 비교", embeddings)]
for t, (_, mod) in zip(st.tabs([n for n, _ in TABS]), TABS):
    with t:
        mod.render(which, source)

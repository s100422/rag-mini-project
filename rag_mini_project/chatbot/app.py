"""금융 문서 질의응답 챗봇. 실행: 저장소 루트에서  streamlit run rag_mini_project/chatbot/app.py

PDF 23개(용어집, 표준약관 15개, 금융생활 가이드북, 추가 문서 6개)에서 근거를 찾아 파일명·쪽과 함께 답한다.
백엔드는 chatbot/core.py(최종 설정: 구조 단위 parent-child 검색 + Agent v2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from chatbot import core  # noqa: E402

EXAMPLES = [
    "휴면성 증권계좌란 무엇인가요?",
    "DB형 퇴직연금의 급여는 어떻게 정해지나요?",
    "피싱으로 돈을 보냈어요. 어떻게 해야 하나요?",
    "이 계좌는 배우자가 이어받을 수 있어?",
    "삼성전자 지금 사도 돼?",
]
SNIPPET = 700  # 근거 원문을 화면에 보이는 최대 글자 수

st.set_page_config(page_title="금융 문서 챗봇", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="검색 엔진을 준비하는 중...")
def backend():
    return core.load()


def snippet(text: str) -> str:
    """원문이 길면 줄 경계에서 자른다(표 한가운데에서 끊기지 않게)."""
    text = text.strip()
    if len(text) <= SNIPPET:
        return text
    cut = text[:SNIPPET]
    return (cut[:cut.rfind("\n")] if "\n" in cut else cut) + "\n\n…(원문 일부)"


def render_reply(reply: core.Reply) -> None:
    if reply.error:
        st.error("답변을 만들지 못했습니다. 잠시 뒤 다시 시도해 주세요.")
        st.caption(reply.error)
        return
    st.markdown(reply.answer)
    meta = " · ".join(x for x in (reply.request_type, f"{reply.seconds:.1f}초", f"모델 호출 {reply.llm_calls}회") if x)
    st.caption(meta)
    if reply.sources:
        cited = sum(s.cited for s in reply.sources)
        with st.expander(f"근거 보기 (인용 {cited}개 / 검색 {len(reply.sources)}개)"):
            for s in sorted(reply.sources, key=lambda s: (not s.cited, s.number)):
                mark = "✅ 인용" if s.cited else "검색됨"
                st.markdown(f"**[{s.number}] {s.file} · {s.pages}** — {s.heading or '(제목 없음)'}  `{mark}`")
                st.markdown("> " + snippet(s.content).replace("\n", "\n> "))


st.title("💬 금융 문서 챗봇")
st.caption("용어집·표준약관·금융생활 가이드북 등 PDF 23개에서 근거를 찾아 답합니다. 투자 자문이 아니며, 문서에 없는 내용은 확인할 수 없다고 답합니다.")

with st.sidebar:
    st.header("예시 질문")
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            st.session_state["pending"] = q
    st.divider()
    if st.button("대화 지우기", width="stretch"):
        st.session_state["history"] = []
        st.rerun()
    st.caption("각 질문은 독립적으로 처리됩니다(앞선 대화를 기억하지 않습니다). 질문에 필요한 정보(상품 종류 등)를 함께 적어 주세요.")

history = st.session_state.setdefault("history", [])
for msg in history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_reply(msg["reply"])

question = st.chat_input("궁금한 금융 내용을 물어보세요") or st.session_state.pop("pending", None)
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 만드는 중..."):
            retriever, run = backend()
            reply = core.ask(question, retriever, run)
        render_reply(reply)
    history.append({"role": "assistant", "reply": reply})

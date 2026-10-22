"""문항 탐색 탭: 두 실행을 골라 문항 단위로 비교한다."""
import pandas as pd
import streamlit as st

from dashboard.data import BEHAVIOR_KO, answer_runs, load_cases, reference_answers
from rag.report import ORDER


def _blocked(r: dict) -> bool:
    return bool(r.get("blocked")) or bool((r.get("answer_metrics") or {}).get("blocked"))


def _pass(r: dict) -> float:
    return float((r.get("answer_metrics") or {}).get("pass") or 0)


def _state(r: dict) -> str:
    return "차단" if _blocked(r) else ("통과" if _pass(r) >= 1 else "실패")


def _pct(v) -> str:
    return "-" if v is None else f"{float(v) * 100:.1f}%"


def _pick(runs: pd.DataFrame, which: str, tag: str, default: str, labels: dict) -> str:
    settings = [s for s in ORDER if s in set(runs.setting)] or sorted(set(runs.setting))
    idx = settings.index(default) if default in settings else 0
    c1, c2 = st.columns(2)
    s = c1.selectbox(f"설정 {tag}", settings, index=idx, format_func=lambda x: labels.get(x, x), key=f"cases_{which}_{tag}_s")
    reps = list(runs[runs.setting == s].sort_values("run_id").rep)
    rep = c2.selectbox(f"실행 {tag}", reps, key=f"cases_{which}_{tag}_{s}_r")
    return runs[(runs.setting == s) & (runs.rep == rep)].sort_values("run_id").iloc[0].run_id


def _detail(col, r: dict | None, tag: str, ref_must: list, ref_not: list) -> None:
    with col:
        st.markdown(f"#### 실행 {tag}")
        if r is None:
            st.info("이 실행에는 해당 문항이 없습니다.")
            return
        m = r.get("answer_metrics") or {}
        out = ((r.get("judge") or {}).get("output")) or {}
        st.markdown(f"**결과: {_state(r)}**" + (f" (사유: {r['blocked']})" if r.get("blocked") else ""))
        if r.get("error"):
            st.error(f"오류: {r['error']}")
        beh = m.get("behavior")
        st.write(f"판정된 행동: {BEHAVIOR_KO.get(beh, beh or '-')}")
        inc = {i.get("index"): i.get("included") for i in out.get("must_include", [])}
        vio = {i.get("index"): i.get("violated") for i in out.get("must_not_include", [])}
        if ref_must:
            st.markdown("**핵심 항목**")
            st.dataframe(pd.DataFrame({"항목": ref_must, "포함": ["-" if inc.get(i) is None else ("✅" if inc[i] else "❌")
                                                                for i in range(1, len(ref_must) + 1)]}), hide_index=True, width="stretch")
        if ref_not:
            st.markdown("**금지 항목**")
            st.dataframe(pd.DataFrame({"항목": ref_not, "위반": ["-" if vio.get(i) is None else ("⚠️" if vio[i] else "아니오")
                                                              for i in range(1, len(ref_not) + 1)]}), hide_index=True, width="stretch")
        st.markdown("**답변**")
        if _blocked(r):
            st.warning("차단됨 (API가 응답을 막아 답변이 비어 있습니다).")
        else:
            st.write(r.get("answer") or "(빈 답변)")
        cites = r.get("citations") or []
        st.write(f"인용 번호: {cites if cites else '없음'}")
        st.markdown("**점수**")
        st.dataframe(pd.DataFrame([{"pass": _state(r), "핵심 항목 포함": _pct(m.get("must_include_rate")), "근거 준수": _pct(m.get("faithfulness")),
                                    "인용 정밀도": _pct(m.get("citation_precision")), "인용 재현율": _pct(m.get("citation_recall"))}]),
                     hide_index=True, width="stretch")
        ret = r.get("retrieved") or []
        if ret:
            st.markdown(f"**컨텍스트 ({len(ret)}개)**")
            st.dataframe(pd.DataFrame([{"순위": x.get("rank"), "인용": "인용" if x.get("rank") in cites else "", "정답 근거": "✅" if x.get("relevant") else "",
                                        "파일": x.get("source_file"), "쪽": x.get("page_start"), "제목": x.get("heading")} for x in ret]),
                         hide_index=True, width="stretch")
            with st.expander("컨텍스트 원문"):
                for x in ret:
                    st.markdown(f"**[{x.get('rank')}] {x.get('source_file')} p.{x.get('page_start')}**" + (" ✅" if x.get("relevant") else ""))
                    st.text(x.get("content") or "")
        u, lat = r.get("usage") or {}, r.get("latency") or {}
        st.dataframe(pd.DataFrame([{"LLM 호출": u.get("llm_calls"), "입력 토큰": u.get("input_tokens"), "출력 토큰": u.get("output_tokens"),
                                    "생각 토큰": u.get("thinking_tokens"), "검색 ms": lat.get("retrieval_ms"), "생성 ms": lat.get("generation_ms"),
                                    "전체 ms": lat.get("total_ms")}]), hide_index=True, width="stretch")
        steps = r.get("steps") or []
        if steps:
            with st.expander("Agent 단계"):
                st.json(steps)


def render(which: str, source: str) -> None:
    runs = answer_runs(source)
    runs = runs[runs.set_name == which] if not runs.empty else runs
    if runs.empty:
        st.warning("이 세트의 답변 단계 실행이 없습니다.")
        return
    labels = runs.drop_duplicates("setting").set_index("setting")["label"].to_dict()
    ca, cb = st.columns(2)
    with ca:
        ida = _pick(runs, which, "A", "B0", labels)
    with cb:
        idb = _pick(runs, which, "B", "E3-b", labels)
    if not st.checkbox("두 실행의 문항별 결과 불러오기", key=f"cases_{which}_go"):
        st.info("선택한 실행을 불러오려면 체크하세요(파일이 큽니다).")
        return
    try:
        A, B = load_cases(source, ida), load_cases(source, idb)
    except Exception as e:  # noqa: BLE001
        st.warning(f"문항 결과를 읽지 못했습니다: {e}")
        return
    ids = [i for i in A if i in B]
    if not ids:
        st.warning("두 실행에 공통 문항이 없습니다.")
        return
    rows = []
    for i in ids:
        a, b = A[i], B[i]
        pa, pb = _pass(a), _pass(b)
        ch = "나아짐" if pa < 1 <= pb else ("나빠짐" if pb < 1 <= pa else "같음")
        rows.append({"case_id": i, "질문": (a.get("question") or "")[:50], "기대 행동": BEHAVIOR_KO.get(a.get("expected_group"), a.get("expected_group") or "-"),
                     "질문 유형": a.get("question_type") or "-", "난이도": a.get("difficulty") or "-",
                     "A pass": _state(a), "B pass": _state(b), "변화": ch})
    df = pd.DataFrame(rows)
    m = st.columns(4)
    m[0].metric("비교 문항", len(df))
    for c, k in zip(m[1:], ["나아짐", "같음", "나빠짐"]):
        c.metric(k, int((df["변화"] == k).sum()))
    st.caption("나아짐 = A 실패 → B 통과, 나빠짐 = A 통과 → B 실패. 같은 문항끼리 맞춘 결과입니다.")

    f = st.columns(4)
    ch = f[0].selectbox("변화", ["전체", "나아짐", "나빠짐", "같음"], key=f"cases_{which}_f1")
    beh = f[1].multiselect("기대 행동", sorted(df["기대 행동"].unique()), key=f"cases_{which}_f2")
    qt = f[2].multiselect("질문 유형", sorted(df["질문 유형"].unique()), key=f"cases_{which}_f3")
    dif = f[3].multiselect("난이도", sorted(df["난이도"].unique()), key=f"cases_{which}_f4")
    v = df
    if ch != "전체":
        v = v[v["변화"] == ch]
    for col, sel in (("기대 행동", beh), ("질문 유형", qt), ("난이도", dif)):
        if sel:
            v = v[v[col].isin(sel)]
    st.dataframe(v, hide_index=True, width="stretch")
    if v.empty:
        st.warning("조건에 맞는 문항이 없습니다.")
        return

    cid = st.selectbox("문항 상세", list(v.case_id), format_func=lambda i: f"{i} — {(A[i].get('question') or '')[:40]}", key=f"cases_{which}_cid")
    a, b = A[cid], B[cid]
    st.divider()
    st.markdown(f"### {cid}")
    st.markdown(f"**질문**: {a.get('question')}")
    st.markdown(f"**기대 행동**: {BEHAVIOR_KO.get(a.get('expected_group'), a.get('expected_group') or '-')}  ·  유형: {a.get('question_type')}  ·  난이도: {a.get('difficulty')}")
    ref = reference_answers().get(cid)
    st.markdown("**기준 답안**")
    st.write(ref if ref else "(없음)")
    must, notm = list(a.get("must_include") or []), list(a.get("must_not_include") or [])
    c1, c2 = st.columns(2)
    _detail(c1, a, "A", must, notm)
    _detail(c2, b, "B", must, notm)

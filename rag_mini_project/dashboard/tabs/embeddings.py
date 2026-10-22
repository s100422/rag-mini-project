"""임베딩 비교 탭: 두 결과 폴더(001·002)를 같은 세트에서 나란히 비교."""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import METRIC_KO, NOISE_PP, RET_LABEL, RET_ORDER, answer_runs, available_sources, retrieval_runs, settings_table

RMETRICS = ["hit@5", "recall@5", "source_coverage@5", "mrr"]
SHORT = lambda s: s.split(" ")[0]  # noqa: E731  'emb-001' / 'emb-2'


def _grouped(df: pd.DataFrame, cat: str, order: list, ytitle: str, tip: str) -> alt.Chart:
    return alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{cat}:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=260, labelOverlap=False)),
        xOffset=alt.XOffset("임베딩:N"), y=alt.Y("value:Q", title=ytitle, scale=alt.Scale(zero=True)),
        color=alt.Color("임베딩:N"), tooltip=[cat, "임베딩", alt.Tooltip("value:Q", format=".1f", title=tip)]).properties(height=380)


def render(which: str, source: str) -> None:
    srcs = available_sources()
    if len(srcs) < 2:
        st.info("임베딩 비교는 결과 폴더 2개(results, results_emb2)가 모두 있어야 합니다.")
        return
    a, b = srcs[0], srcs[1]
    na, nb = SHORT(a), SHORT(b)
    st.warning("gemini-embedding-2는 task_type(질의·문서 구분)을 무시한다. 질의 벡터와 문서 벡터의 코사인이 1.000이다(001은 0.885). "
               "그래서 002 결과는 접두어 없이 쓴 경우의 값이며 **002의 최대 성능이 아니다.** 근거: docs/experiment_results_emb2.md 0절.")

    # 1) 검색 단계
    st.subheader("검색 단계")
    ra, rb = retrieval_runs(a), retrieval_runs(b)
    ra, rb = (r[r.set_name == which] if not r.empty else r for r in (ra, rb))
    if ra.empty or rb.empty:
        st.warning("이 세트의 검색 단계 실행이 한쪽 또는 양쪽에 없습니다.")
    else:
        metric = st.radio("지표", RMETRICS, horizontal=True, key="emb_ret_metric")
        both = [k for k in RET_ORDER if k in set(ra.key) and k in set(rb.key)]
        va, vb = ra.set_index("key")[metric] * 100, rb.set_index("key")[metric] * 100
        long = pd.DataFrame([{"실험": RET_LABEL[k], "임베딩": n, "value": v[k]} for k in both for n, v in ((na, va), (nb, vb))])
        order = [RET_LABEL[k] for k in both]
        st.altair_chart(_grouped(long, "실험", order, f"{metric} (%)", metric), width="stretch")
        tab = pd.DataFrame({"실험": order, na: [va[k] for k in both], nb: [vb[k] for k in both]})
        tab["Δ(%p) = 002 − 001"] = tab[nb] - tab[na]
        st.dataframe(tab.round(1), hide_index=True, width="stretch")
        st.caption("E5-a(BM25)는 임베딩을 쓰지 않아 두 폴더 값이 같다. 대조군으로, 다르면 평가 설정이 어긋난 것이다.")

    # 3) 답변 단계
    st.subheader("답변 단계")
    aa, ab = answer_runs(a), answer_runs(b)
    aa, ab = (r[r.set_name == which] if not r.empty else r for r in (aa, ab))
    if aa.empty or ab.empty:
        st.warning("이 세트의 답변 단계 실행이 한쪽 또는 양쪽에 없습니다.")
    else:
        amet = st.radio("답변 지표", ["pass", "must_include_rate"], horizontal=True, format_func=METRIC_KO.get, key="emb_ans_metric")
        ta = settings_table(aa, amet).set_index("setting")
        tb = settings_table(ab, amet).set_index("setting")
        common = [s for s in ta.index if s in tb.index]
        if not common:
            st.warning("두 폴더에 모두 있는 설정이 없습니다.")
        else:
            labels = [ta.loc[s, "label"] for s in common]
            long = pd.DataFrame([{"설정": ta.loc[s, "label"], "임베딩": n, "value": t.loc[s, "mean"] * 100}
                                 for s in common for n, t in ((na, ta), (nb, tb))])
            st.altair_chart(_grouped(long, "설정", labels, f"{METRIC_KO[amet]} (%, 반복 평균)", "평균"), width="stretch")
            w = NOISE_PP[amet]
            d = [(tb.loc[s, "mean"] - ta.loc[s, "mean"]) * 100 for s in common]
            tab = pd.DataFrame({"설정": labels, f"{na} 평균 %": [ta.loc[s, "mean"] * 100 for s in common],
                                f"{nb} 평균 %": [tb.loc[s, "mean"] * 100 for s in common],
                                "차이(%p) = 002 − 001": d,
                                "판정": ["차이 없음" if abs(x) <= w else ("002 높음" if x > 0 else "001 높음") for x in d],
                                f"반복 수 {na}/{nb}": [f"{int(ta.loc[s, 'runs'])}/{int(tb.loc[s, 'runs'])}" for s in common]})
            st.dataframe(tab.round(1), hide_index=True, width="stretch")
            st.caption(f"반복 오차 안쪽(±{w:.0f}%p 이하)이면 '차이 없음'으로 본다. 반복 평균이며 반복 수가 다르면 신뢰도가 다르다.")

    # 4) B0 대비 개선폭
    st.subheader("각 임베딩 안에서 B0 대비 개선폭")
    st.caption("청킹 효과(예: E3-b − B0)가 임베딩이 바뀌어도 유지되는지 본다. 값은 %p.")
    if ra.empty or rb.empty:
        return
    rm = st.radio("검색 지표", RMETRICS, horizontal=True, key="emb_gain_metric")
    ia, ib = ra.set_index("key")[rm] * 100, rb.set_index("key")[rm] * 100
    if "B0" not in ia.index or "B0" not in ib.index:
        st.warning("한쪽에 B0 검색 실행이 없어 개선폭을 계산할 수 없습니다.")
        return
    keys = [k for k in RET_ORDER if k != "B0" and k in ia.index and k in ib.index]
    out = pd.DataFrame({"실험": [RET_LABEL[k] for k in keys], f"{na} 개선폭": [ia[k] - ia["B0"] for k in keys],
                        f"{nb} 개선폭": [ib[k] - ib["B0"] for k in keys]})
    out["같은 방향?"] = ["예" if (x > 0) - (x < 0) == (y > 0) - (y < 0) else "아니오" for x, y in zip(out.iloc[:, 1], out.iloc[:, 2])]
    st.dataframe(out.round(1), hide_index=True, width="stretch")

"""검색 단계 탭: 실험별 검색 지표, B0 대비 표, k 곡선."""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import RET_LABEL, RET_ORDER, retrieval_runs

METRICS = ["hit@5", "recall@5", "source_coverage@5", "mrr", "dup_rate@5", "context_chars@5"]
PCT = {m: m != "context_chars@5" for m in METRICS}  # context_chars만 글자 수
UNIT_NOTE = {"hit@5": "정답 근거가 상위 5개 안에 하나라도 있는 비율", "recall@5": "정답 근거 중 상위 5개에 들어온 비율",
             "source_coverage@5": "정답 근거 문서 중 상위 5개가 덮은 비율", "mrr": "첫 정답 순위의 역수 평균(높을수록 앞에 나옴)",
             "dup_rate@5": "상위 5개 안의 중복 비율(낮을수록 좋음)",
             "context_chars@5": "모델에 들어가는 컨텍스트 글자 수. 작을수록 좋음(비용·잡음 감소)"}
# 기준이 B0가 아닌 실험
BASE_OF = {"E3-b": "E3-a"}


def _scale(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    if PCT[metric]:
        df[metric] = df[metric] * 100
    return df


def render(which: str, source: str) -> None:
    runs = retrieval_runs(source)
    runs = runs[runs.set_name == which] if not runs.empty else runs
    if runs.empty:
        st.warning("이 세트의 검색 단계 실행이 없습니다.")
        return
    runs = runs.set_index("key").reindex([k for k in RET_ORDER if k in set(runs.key)]).reset_index()
    runs["label"] = runs["key"].map(RET_LABEL)
    order = list(runs.label)

    metric = st.radio("지표", METRICS, horizontal=True, key="ret_metric")
    st.caption(f"{metric}: {UNIT_NOTE[metric]}.")
    unit = "%" if PCT[metric] else "자"
    fmt = ".1f" if PCT[metric] else ",.0f"
    d = _scale(runs, metric)
    d = d[["key", "label", "method", metric]].rename(columns={metric: "value"})

    x = alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=260, labelOverlap=False))
    y = alt.Y("value:Q", title=f"{metric} ({unit})", scale=alt.Scale(zero=True))
    layers = [alt.Chart(d).mark_bar(opacity=0.7).encode(
        x=x, y=y, tooltip=["label", "method", alt.Tooltip("value:Q", format=fmt, title=metric)])]
    b0 = d.loc[d.key == "B0", "value"]
    if len(b0):
        layers.append(alt.Chart(pd.DataFrame({"v": [float(b0.iloc[0])]})).mark_rule(strokeDash=[4, 4], color="#888").encode(y="v:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=380), width="stretch")
    st.caption("회색 점선 = B0 값." + (" 막대가 낮을수록 좋다." if metric in ("context_chars@5", "dup_rate@5") else ""))

    # B0 대비 표
    st.subheader("기준 대비 차이")
    st.caption("기준은 B0이고, E3-b만 E3-a(구조 단위)를 기준으로 한다(parent-child 효과만 보려고).")
    val = d.set_index("key")["value"]
    rows = []
    for r in d.itertuples():
        if r.key == "B0":
            rows.append({"실험": r.key, "바꾼 것": r.label, "기준": "-", metric: r.value, "차이": 0.0})
            continue
        ref = BASE_OF.get(r.key, "B0")
        diff = r.value - val[ref] if ref in val.index else float("nan")
        rows.append({"실험": r.key, "바꾼 것": r.label, "기준": ref, metric: r.value, "차이": diff})
    tab = pd.DataFrame(rows)
    dlabel = "차이(%p)" if PCT[metric] else "차이(자)"
    tab = tab.rename(columns={"차이": dlabel})
    st.dataframe(tab.round(1) if PCT[metric] else tab.round(0), hide_index=True, width="stretch")

    # k 곡선
    st.subheader("k에 따른 변화")
    kmetric = st.radio("곡선 지표", ["hit", "recall"], horizontal=True, key="ret_kmetric", format_func=lambda m: f"{m}@k")
    default = [k for k in ("B0", "E3-a", "E3-b") if k in set(runs.key)]
    picked = st.multiselect("실험", list(runs.key), default=default, format_func=lambda k: RET_LABEL.get(k, k), key="ret_ks")
    if picked:
        ks = [1, 3, 5, 10]
        recs = []
        for _, r in runs[runs.key.isin(picked)].iterrows():
            for k in ks:
                recs.append({"label": r["label"], "k": k, "value": r[f"{kmetric}@{k}"] * 100})
        long = pd.DataFrame(recs)
        line = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("k:O", title="k"), y=alt.Y("value:Q", title=f"{kmetric}@k (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("label:N", title="실험", sort=order),
            tooltip=["label", "k", alt.Tooltip("value:Q", format=".1f")])
        st.altair_chart(line.properties(height=340), width="stretch")
    else:
        st.info("실험을 하나 이상 고르세요.")
    st.caption("E6-a·b(컨텍스트 k=3·10)의 검색 지표는 B0 실행의 k=3·10 값으로 읽는다(검색은 같고 모델에 넘기는 개수만 다르다).")

    st.info("읽는 법: 확장 세트는 hit가 천장(95% 안팎)에 가까워 실험 간 차이가 작다. recall·source_coverage·컨텍스트 양을 같이 봐야 한다. "
            "기본 세트는 hit만으로도 차이가 드러난다.")

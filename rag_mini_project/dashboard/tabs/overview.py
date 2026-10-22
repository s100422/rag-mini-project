"""개요 탭: 설정별 pass·핵심 항목 포함(반복 평균, 최소–최대, 개별 실행 점)과 B0 기준선·'차이 없음' 띠."""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import METRIC_KO, NOISE_PP, answer_runs, settings_table


def render(which: str, source: str) -> None:
    runs = answer_runs(source)
    runs = runs[runs.set_name == which]
    if runs.empty:
        st.warning("이 세트의 답변 단계 실행이 없습니다.")
        return
    metric = st.radio("지표", ["pass", "must_include_rate"], format_func=METRIC_KO.get, horizontal=True)
    tab = settings_table(runs, metric)
    for c in ("mean", "min", "max"):
        tab[c] = tab[c] * 100
    pts = runs[["setting", "label", metric]].assign(value=lambda d: d[metric] * 100)
    base = tab.loc[tab.setting == "B0", "mean"]
    order = list(tab.label)
    y = alt.Y("value:Q", title=f"{METRIC_KO[metric]} (%)", scale=alt.Scale(zero=False))
    x = alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=260, labelOverlap=False))
    layers = []
    if len(base):
        b, w = float(base.iloc[0]), NOISE_PP[metric]
        band = pd.DataFrame({"lo": [b - w], "hi": [b + w]})
        layers.append(alt.Chart(band).mark_rect(opacity=0.12, color="#888").encode(y="lo:Q", y2="hi:Q"))
        layers.append(alt.Chart(pd.DataFrame({"v": [b]})).mark_rule(strokeDash=[4, 4], color="#888").encode(y="v:Q"))
    bars = alt.Chart(tab.rename(columns={"mean": "value"})).mark_bar(opacity=0.55).encode(
        x=x, y=y, tooltip=["label", alt.Tooltip("value:Q", format=".1f", title="평균"), alt.Tooltip("min:Q", format=".1f"),
                           alt.Tooltip("max:Q", format=".1f"), alt.Tooltip("runs:Q", title="반복 수")])
    rng = alt.Chart(tab).mark_rule(strokeWidth=2).encode(x=x, y=alt.Y("min:Q"), y2="max:Q")
    dots = alt.Chart(pts).mark_circle(size=60, color="black").encode(x=x, y=y, tooltip=["label", alt.Tooltip("value:Q", format=".1f")])
    st.altair_chart(alt.layer(*layers, bars, rng, dots).properties(height=420), width="stretch")
    st.caption(f"막대 = 반복 평균, 세로선 = 최소–최대, 점 = 개별 실행. 회색 점선 = B0 평균, 회색 띠 = B0 ±{NOISE_PP[metric]:.0f}%p "
               "(이 안쪽 차이는 '차이 없음'으로 본다, 계획 4.3절).")
    show = tab.assign(차이_B0=lambda d: d["mean"] - float(base.iloc[0]) if len(base) else None)
    st.dataframe(show.rename(columns={"label": "설정", "runs": "실행 수", "mean": "평균 %", "min": "최소 %", "max": "최대 %", "차이_B0": "B0 대비 %p"})
                 .drop(columns="setting").round(1), hide_index=True, width="stretch")

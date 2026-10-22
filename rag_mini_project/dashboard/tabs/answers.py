"""답변 비교 탭: 설정 × 지표 막대, 그룹별 히트맵, 전체 지표 표, 해석 가이드."""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import BEHAVIOR_KO, GROUP_KO, METRIC_KO, NOISE_PP, answer_groups, answer_runs
from rag.report import LABEL, ORDER

BAR_METRICS = ["pass", "must_include_rate", "coarse_match", "faithfulness", "citation_precision", "citation_recall"]
HEAT_GROUPS = ["expected_group", "difficulty", "category", "question_type", "needs_table", "n_sources", "source_doc"]
HEAT_METRICS = ["pass", "must_include_rate"]
DEFAULT_SETTINGS = ["B0", "E3-b", "C0", "A2", "A4"]


def _group_label(gtype: str, g: str) -> str:
    return BEHAVIOR_KO.get(g, g) if gtype == "expected_group" else str(g)


def _ordered(pick: list[str]) -> list[str]:
    return [s for s in ORDER if s in pick]


def render(which: str, source: str) -> None:
    runs = answer_runs(source)
    runs = runs[runs.set_name == which] if not runs.empty else runs
    if runs.empty:
        st.warning("이 세트의 답변 단계 실행이 없습니다.")
        return
    present = [s for s in ORDER if s in set(runs.setting)]
    pick = st.multiselect("설정", present, default=[s for s in DEFAULT_SETTINGS if s in present],
                          format_func=lambda s: LABEL.get(s, s), key=f"ans_settings_{which}")
    if not pick:
        st.warning("설정을 하나 이상 고르세요.")
        return
    order = _ordered(pick)
    sel = runs[runs.setting.isin(pick)]

    # 1. 설정 × 지표 묶음 막대
    st.subheader("설정별 지표")
    metrics = st.multiselect("지표", BAR_METRICS, default=["pass", "must_include_rate"], format_func=METRIC_KO.get,
                             key=f"ans_metrics_{which}")
    if not metrics:
        st.warning("지표를 하나 이상 고르세요.")
    else:
        m = sel.groupby("setting")[metrics].mean().reset_index().melt("setting", var_name="metric", value_name="v").dropna()
        m["v"] = m["v"] * 100
        m["지표"] = m["metric"].map(METRIC_KO)
        m["설정"] = m["setting"]
        if m.empty:
            st.warning("선택한 지표 값이 없습니다.")
        else:
            chart = alt.Chart(m).mark_bar().encode(
                x=alt.X("설정:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("지표:N"), y=alt.Y("v:Q", title="반복 평균 (%)"), color=alt.Color("지표:N"),
                tooltip=["설정", "지표", alt.Tooltip("v:Q", format=".1f", title="평균 %")]).properties(height=380)
            txt = alt.Chart(m).mark_text(dy=-6, fontSize=10).encode(
                x=alt.X("설정:N", sort=order), xOffset="지표:N", y="v:Q", text=alt.Text("v:Q", format=".1f"))
            st.altair_chart(chart + txt, width="stretch")
            st.caption("막대 = 같은 설정을 반복 실행한 평균. " + " · ".join(f"{s} = {LABEL[s]}" for s in order))

    # 2. 그룹별 히트맵
    st.subheader("그룹별 히트맵")
    c1, c2 = st.columns(2)
    gtype = c1.selectbox("그룹 기준", HEAT_GROUPS, format_func=GROUP_KO.get, key=f"ans_gtype_{which}")
    hm = c2.selectbox("지표", HEAT_METRICS, format_func=METRIC_KO.get, key=f"ans_hmetric_{which}")
    gr = answer_groups(source)
    gr = gr[(gr.set_name == which) & (gr.group_type == gtype) & gr.setting.isin(pick)] if not gr.empty else gr
    if gr.empty:
        st.warning("이 기준의 그룹 결과가 없습니다.")
    else:
        h = gr.groupby(["setting", "group"]).agg(v=(hm, "mean"), n=("n", "first")).reset_index().dropna(subset=["v"])
        h["v"] = h["v"] * 100
        h["그룹"] = [f"{_group_label(gtype, g)} (n={n})" for g, n in zip(h["group"], h["n"])]
        h["설정"] = h["setting"]
        cols = list(h.sort_values(["n", "group"], ascending=[False, True])["그룹"].drop_duplicates())
        base = alt.Chart(h).encode(x=alt.X("그룹:N", sort=cols, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=200)),
                                   y=alt.Y("설정:N", sort=order, title=None))
        rect = base.mark_rect().encode(color=alt.Color("v:Q", scale=alt.Scale(domain=[0, 100], scheme="blues"), title="%"),
                                       tooltip=["설정", "그룹", alt.Tooltip("v:Q", format=".1f", title="평균 %")])
        text = base.mark_text(fontSize=11).encode(text=alt.Text("v:Q", format=".0f"),
                                                  color=alt.condition("datum.v > 60", alt.value("white"), alt.value("black")))
        st.altair_chart((rect + text).properties(height=max(120, 36 * len(pick) + 40)), width="stretch")
        st.caption(f"칸 = {METRIC_KO[hm]} 반복 평균(%), 열 머리의 n = 문항 수. 그룹이 작으면 한두 문항에 값이 크게 흔들린다"
                   "(기본 세트: 되묻기 8·정정 7·보류 12문항).")

    # 3. 표
    st.subheader("선택한 설정의 전체 지표")
    g = sel.groupby("setting")
    out_tok = sel.assign(o=sel.output_tokens_mean.fillna(0) + sel.thinking_tokens_mean.fillna(0)).groupby("setting")["o"].mean()
    t = pd.DataFrame({"실행 수": g.size(), "pass %": g["pass"].mean() * 100, "포함 %": g["must_include_rate"].mean() * 100,
                      "일치 %": g["coarse_match"].mean() * 100, "근거 준수 %": g["faithfulness"].mean() * 100,
                      "인용 정밀도 %": g["citation_precision"].mean() * 100, "인용 재현율 %": g["citation_recall"].mean() * 100,
                      "입력 토큰/문항": g["input_tokens_mean"].mean(), "출력 토큰/문항": out_tok,
                      "LLM 호출": g["llm_calls_mean"].mean(), "응답 p50(초)": g["total_ms_p50"].mean() / 1000})
    t = t.reindex(order).reset_index().rename(columns={"setting": "설정"})
    t.insert(1, "이름", t["설정"].map(LABEL))
    st.dataframe(t.round(1), hide_index=True, width="stretch")

    # 4. 해석 가이드
    st.caption(f"해석: B0 대비 pass {NOISE_PP['pass']:.0f}%p, 핵심 항목 포함 {NOISE_PP['must_include_rate']:.0f}%p 안쪽 차이는 "
               "같은 설정을 다시 돌려도 나오는 흔들림이라 '차이 없음'으로 본다.")
    st.caption("'근거 준수'가 Agent 설정에서 낮게 나오는 것은 실제 환각이 늘어서가 아니라, 거절할 때 붙인 위험 안내를 "
               "'근거 없는 주장'으로 세는 지표 성격 때문이다.")

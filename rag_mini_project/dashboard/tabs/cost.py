"""비용 대비 품질 탭: 비용-품질 산점도와 'B0 대비 품질 1%p당 추가 비용' 표."""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import METRIC_KO, NOISE_PP, answer_runs
from rag.report import LABEL, ORDER

X_OPTS = {"input_tokens_mean": "입력 토큰/문항", "llm_calls_mean": "LLM 호출 수", "p50_s": "응답 p50(초)"}
STAGE_KO = {"rag": "RAG", "agent": "Agent"}


def _agg(runs: pd.DataFrame) -> pd.DataFrame:
    r = runs.assign(p50_s=runs.total_ms_p50 / 1000)
    g = r.groupby("setting")
    d = pd.DataFrame({"stage": g["stage"].first(), "runs": g.size(),
                      "input_tokens_mean": g["input_tokens_mean"].mean(), "llm_calls_mean": g["llm_calls_mean"].mean(),
                      "p50_s": g["p50_s"].mean()})
    for m in ("pass", "must_include_rate"):
        d[m], d[m + "_min"], d[m + "_max"] = g[m].mean() * 100, g[m].min() * 100, g[m].max() * 100
    d = d.reindex([s for s in ORDER if s in d.index]).reset_index()
    d["구분"] = d["stage"].map(STAGE_KO)
    return d


def render(which: str, source: str) -> None:
    runs = answer_runs(source)
    runs = runs[runs.set_name == which] if not runs.empty else runs
    if runs.empty:
        st.warning("이 세트의 답변 단계 실행이 없습니다.")
        return
    a = _agg(runs)
    c1, c2 = st.columns(2)
    xk = c1.selectbox("x축(비용)", list(X_OPTS), format_func=X_OPTS.get, key=f"cost_x_{which}")
    ym = c2.selectbox("y축(품질)", ["pass", "must_include_rate"], format_func=METRIC_KO.get, key=f"cost_y_{which}")
    d = a.dropna(subset=[xk, ym]).copy()
    if d.empty:
        st.warning("선택한 축의 값이 없습니다.")
        return
    d["y"], d["ymin"], d["ymax"], d["x"] = d[ym], d[ym + "_min"], d[ym + "_max"], d[xk]
    color = alt.Color("구분:N", scale=alt.Scale(domain=["RAG", "Agent"], range=["#4c78a8", "#e45756"]), title="구분")
    enc = dict(x=alt.X("x:Q", title=X_OPTS[xk], scale=alt.Scale(zero=False)),
               y=alt.Y("y:Q", title=f"{METRIC_KO[ym]} (%)", scale=alt.Scale(zero=False)))
    tip = ["setting", alt.Tooltip("구분:N"), alt.Tooltip("x:Q", format=",.1f", title=X_OPTS[xk]),
           alt.Tooltip("y:Q", format=".1f", title="평균 %"), alt.Tooltip("ymin:Q", format=".1f", title="최소"),
           alt.Tooltip("ymax:Q", format=".1f", title="최대"), alt.Tooltip("llm_calls_mean:Q", format=".1f", title="LLM 호출"),
           alt.Tooltip("runs:Q", title="실행 수")]
    err = alt.Chart(d).mark_rule(opacity=0.6).encode(x="x:Q", y="ymin:Q", y2="ymax:Q", color=color)
    pts = alt.Chart(d).mark_circle(opacity=0.85).encode(
        **enc, color=color, size=alt.Size("llm_calls_mean:Q", scale=alt.Scale(range=[80, 700]), title="LLM 호출 수"), tooltip=tip)
    b0 = alt.Chart(d[d.setting == "B0"]).mark_point(shape="circle", size=900, filled=False, color="black", strokeWidth=2).encode(**enc)
    txt = alt.Chart(d).mark_text(dx=10, dy=-10, align="left", fontSize=11).encode(**enc, text="setting:N")
    st.altair_chart(alt.layer(err, pts, b0, txt).properties(height=460), width="stretch")
    st.caption("점 = 설정의 반복 평균, 세로선 = 최소–최대, 점 크기 = LLM 호출 수, 검은 테두리 = B0 기준점. "
               "파란색 = RAG, 빨간색 = Agent. 왼쪽 위일수록 싸고 좋다.")

    st.subheader("품질 1%p를 올리는 데 든 추가 비용 (B0 대비)")
    q = st.radio("품질 기준", ["pass", "must_include_rate"], format_func=METRIC_KO.get, horizontal=True, key=f"cost_q_{which}")
    if "B0" not in set(a.setting):
        st.warning("이 세트에 B0 기준 실행이 없어 비교할 수 없습니다.")
        return
    b = a[a.setting == "B0"].iloc[0]
    t = a[a.setting != "B0"].copy()
    if t.empty:
        st.warning("B0 외 설정이 없습니다.")
        return
    t["dq"] = t[q] - b[q]
    t["dtok"] = t["input_tokens_mean"] - b["input_tokens_mean"]
    t["dcall"] = t["llm_calls_mean"] - b["llm_calls_mean"]
    t["dsec"] = t["p50_s"] - b["p50_s"]
    w = NOISE_PP[q]

    def verdict(dq):
        if pd.isna(dq) or dq <= 0:
            return "효과 없음"
        return "이득이 오차 범위" if dq <= w else "의미 있는 개선"

    def per_pp(r):
        if pd.isna(r.dq) or r.dq <= 0:
            return "효과 없음"
        return f"{r.dtok / r.dq:,.0f}" if r.dtok > 0 else "추가 비용 없음"

    out = pd.DataFrame({"설정": t["setting"], "구분": t["구분"], "이름": t["setting"].map(LABEL),
                        f"{METRIC_KO[q]} 차이(%p)": t["dq"].round(1), "추가 입력 토큰": t["dtok"].round(0),
                        "추가 LLM 호출": t["dcall"].round(2), "추가 응답 시간(초)": t["dsec"].round(1),
                        "1%p당 추가 토큰": t.apply(per_pp, axis=1), "판정": t["dq"].map(verdict)})
    st.dataframe(out, hide_index=True, width="stretch")
    st.caption(f"B0: {METRIC_KO[q]} {b[q]:.1f}%, 입력 {b['input_tokens_mean']:,.0f} 토큰, LLM 호출 {b['llm_calls_mean']:.1f}회, "
               f"응답 p50 {b['p50_s']:.1f}초. 품질 차이가 {w:.0f}%p 이하면 '이득이 오차 범위', 0 이하면 '효과 없음'으로 두고 나누지 않는다.")
    st.caption("Agent(빨강)가 비슷한 비용의 RAG(파랑)보다 위에 있는지가 비교 포인트다. 오른쪽으로 갈수록 비싸다.")

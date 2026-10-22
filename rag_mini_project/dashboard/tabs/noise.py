"""반복 오차 탭: 같은 설정을 여러 번 돌렸을 때의 흔들림과 'A vs B' 판정기."""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import METRIC_KO, NOISE_PP, answer_runs, settings_table

METRICS = ["pass", "must_include_rate"]


def _spread(runs: pd.DataFrame, setting: str, metric: str):
    v = runs.loc[runs.setting == setting, metric].dropna() * 100
    return (float(v.max() - v.min()), len(v)) if len(v) else (None, 0)


def _verdict(a: pd.Series, b: pd.Series, noise: float) -> tuple[float, bool, str]:
    """(평균 차이 %p, 범위 겹침, 결론). a, b는 0~1 값."""
    a, b = a.dropna() * 100, b.dropna() * 100
    diff = float(b.mean() - a.mean())
    overlap = bool(max(a.min(), b.min()) <= min(a.max(), b.max()))
    if abs(diff) <= noise:
        return diff, overlap, "차이 없음(오차 범위 안)"
    if not overlap:
        return diff, overlap, "차이 있음(오차 범위 밖, 범위도 겹치지 않음)"
    return diff, overlap, "경계(차이는 기준 밖이나 범위가 겹침)"


def render(which: str, source: str) -> None:
    runs = answer_runs(source)
    runs = runs[runs.set_name == which] if not runs.empty else runs
    if runs.empty:
        st.warning("이 세트의 답변 단계 실행이 없습니다.")
        return
    counts = runs.groupby("setting").size()
    rep_settings = set(counts[counts >= 2].index)

    # 3. 전체 요약
    st.subheader("기준 B0의 반복 폭")
    cols = st.columns(len(METRICS))
    for c, m in zip(cols, METRICS):
        sp, n = _spread(runs, "B0", m)
        if sp is None or n < 2:
            c.metric(f"{METRIC_KO[m]} 폭 (B0)", "-", help="B0 반복이 2회 미만입니다.")
        else:
            v = runs.loc[runs.setting == "B0", m] * 100
            c.metric(f"{METRIC_KO[m]} 폭 (B0, {n}회)", f"{sp:.1f}%p", help=f"최소 {v.min():.1f}% ~ 최대 {v.max():.1f}%. 기준 {NOISE_PP[m]:.0f}%p")
    st.caption("같은 설정·같은 입력인데도(temperature 0) 생성·채점 때문에 점수가 흔들립니다. 이 폭보다 작은 차이는 설정 효과로 보기 어렵습니다.")

    # 1. 반복 설정 점 차트·표
    st.subheader("설정별 반복 실행")
    rr = runs[runs.setting.isin(rep_settings)]
    if rr.empty:
        st.warning("반복이 2회 이상인 설정이 없습니다.")
    else:
        order = list(settings_table(rr, "pass").label)
        cc = st.columns(len(METRICS))
        for c, m in zip(cc, METRICS):
            pts = rr[["setting", "label", "rep", m]].assign(value=lambda d, m=m: d[m] * 100, B0=lambda d: d.setting == "B0")
            ch = alt.Chart(pts).mark_circle(size=90, opacity=0.8).encode(
                y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("value:Q", title=f"{METRIC_KO[m]} (%)", scale=alt.Scale(zero=False)),
                color=alt.Color("B0:N", scale=alt.Scale(domain=[True, False], range=["#d62728", "#4c78a8"]), legend=None),
                tooltip=["label", "rep", alt.Tooltip("value:Q", format=".1f", title=METRIC_KO[m])]).properties(
                height=max(160, 40 * len(order)), title=METRIC_KO[m])
            c.altair_chart(ch, width="stretch")
        st.caption("점 하나 = 실행 1회. 빨간색 = B0(기준).")
        parts = []
        for m in METRICS:
            t = settings_table(rr, m)
            t = t.assign(**{c: t[c] * 100 for c in ("mean", "min", "max")})
            t["폭"] = t["max"] - t["min"]
            t.columns = ["setting", "설정", "실행 수", f"{METRIC_KO[m]} 평균", f"{METRIC_KO[m]} 최소", f"{METRIC_KO[m]} 최대", f"{METRIC_KO[m]} 폭(%p)"]
            parts.append(t if not parts else t.drop(columns=["설정", "실행 수"]))
        tab = parts[0].merge(parts[1], on="setting").drop(columns="setting")
        tab["설정"] = tab["설정"].where(~tab["설정"].str.startswith("B0"), "★ " + tab["설정"])
        st.dataframe(tab.round(1), hide_index=True, width="stretch")

    # 2. 판정기
    st.subheader("A vs B 판정기")
    labels = runs.drop_duplicates("setting").set_index("setting")["label"].to_dict()
    opts = list(settings_table(runs, "pass").setting)
    ca, cb = st.columns(2)
    sa = ca.selectbox("설정 A", opts, index=opts.index("B0") if "B0" in opts else 0, format_func=labels.get, key="noise_a")
    sb = cb.selectbox("설정 B", opts, index=min(1, len(opts) - 1), format_func=labels.get, key="noise_b")
    for s in (sa, sb):
        if counts[s] < 2:
            st.warning(f"{labels[s]}: 반복 1회라 판단 어려움 (범위를 알 수 없어 평균 차이만 참고하세요).")
    rows, concl = [], []
    for m in METRICS:
        a, b = runs.loc[runs.setting == sa, m], runs.loc[runs.setting == sb, m]
        diff, overlap, verdict = _verdict(a, b, NOISE_PP[m])
        if min(counts[sa], counts[sb]) < 2:
            verdict = "반복 1회라 판단 어려움"
        rows.append({"지표": METRIC_KO[m], "A 평균": a.mean() * 100, "B 평균": b.mean() * 100, "차이 B-A (%p)": diff,
                     "A 범위": f"{a.min()*100:.1f}~{a.max()*100:.1f}", "B 범위": f"{b.min()*100:.1f}~{b.max()*100:.1f}",
                     "범위 겹침": "예" if overlap else "아니오", "기준(%p)": NOISE_PP[m],
                     "기준 안쪽": "예" if abs(diff) <= NOISE_PP[m] else "아니오", "결론": verdict})
        concl.append(f"- {METRIC_KO[m]}: {verdict}")
    st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, width="stretch")
    st.markdown("**결론**\n" + "\n".join(concl))

"""README용 결과 그래프 4장을 만든다. 001 임베딩 결과(results/)만 읽는다.

실행: rag_mini_project/ 에서  python docs/make_figures.py   (matplotlib, pandas 필요)
출력: docs/img/fig_*.png
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dashboard import data as D  # noqa: E402  (streamlit 경고는 무시)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
SRC = next(iter(D.SOURCES))  # emb-001
BLUE, BLUE_D, BLUE_L = "#1f77b4", "#0b3d91", "#9ecae1"
RED, RED_D, RED_L = "#d62728", "#8b0000", "#f4a6a6"
KEY = {"E3-b", "C0", "A4"}
SETS = [("base", "기본 세트 (용어집+약관, 100문항)"), ("ext", "확장 세트 (가이드북+추가 문서, 80문항)")]
SHORT = {"B0": "B0 기준", "E3-a": "E3-a 구조 단위", "E3-b": "E3-b 구조+parent-child", "E4-a": "E4-a 제목 접두어",
         "E4-b": "E4-b 표 보존", "E6-a": "E6-a k=3", "E6-b": "E6-b k=10", "E8a": "E8a E3-b+k=3", "E8b": "E8b E3-b+k=10",
         "E7": "E7 Agent 전체", "E7a": "E7a Agent 재검색 없음", "C0": "C0 E3-b+프롬프트 v2",
         "A1": "A1 E3-b+Agent", "A2": "A2 E3-b+Agent v2", "A4": "A4 Agent v2+프롬프트 v2"}
ANS_ORDER = ["B0", "E3-a", "E3-b", "E4-a", "E4-b", "E6-a", "E6-b", "E8a", "E8b", "E7", "E7a", "C0", "A1", "A2", "A4"]


def color(setting, stage):
    agent = stage == "agent"
    if setting in KEY or setting == "B0":
        return RED_D if agent else BLUE_D
    return RED_L if agent else BLUE_L


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


runs = D.answer_runs(SRC)
runs = runs[runs.setting.isin(ANS_ORDER)]

# ---------- 1. 검색 hit@5 ----------
ret = D.retrieval_runs(SRC)
fig, axes = plt.subplots(2, 1, figsize=(10, 9))
for ax, (sn, title) in zip(axes, SETS):
    title = title.replace("100문항", "검색 평가 68문항") if sn == "base" else title  # 검색 평가는 근거가 있는 68문항만
    sub = ret[ret.set_name == sn].set_index("key")
    keys = [k for k in D.RET_ORDER if k in sub.index]
    vals = [sub.loc[k, "hit@5"] * 100 for k in keys]
    cols = [("#444444" if k == "B0" else BLUE_D if k in ("E3-a", "E3-b") else BLUE if k.startswith("E5") or k == "E6-c" else BLUE_L) for k in keys]
    bars = ax.bar(keys, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v + 1e-6:.1f}", ha="center", va="bottom", fontsize=9,
                bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9), zorder=5)
    b0 = sub.loc["B0", "hit@5"] * 100
    ax.axhline(b0, ls="--", color="k", lw=1, label=f"B0 기준 {b0 + 1e-6:.1f}%")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("hit@5 (%)")
    ax.set_xlabel("실험 (E5-a=BM25, E5-b=hybrid, E6-c=MMR)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("검색 단계 hit@5 (정답 근거가 상위 5개에 있는 비율)", fontsize=14, y=0.995)
fig.tight_layout()
save(fig, "fig_retrieval_hit.png")

# ---------- 2. 답변 pass ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 13))
for ax, (sn, title) in zip(axes, SETS):
    r = runs[runs.set_name == sn]
    tab = D.settings_table(r, "pass").set_index("setting")
    order = [s for s in ANS_ORDER if s in tab.index]
    stage = r.groupby("setting").stage.first()
    b0 = tab.loc["B0", "mean"] * 100
    ax.axvspan(b0 - 5, b0 + 5, color="#dddddd", alpha=0.7, label="B0 ±5%p (차이 없음 범위)", zorder=0)
    ax.axvline(b0, ls="--", color="k", lw=1, label=f"B0 평균 {b0:.1f}%")
    for i, s in enumerate(order):
        m, lo, hi = (tab.loc[s, c] * 100 for c in ("mean", "min", "max"))
        ax.barh(i, m, color=color(s, stage[s]), height=0.62, zorder=2)
        ax.plot([lo, hi], [i, i], color="k", lw=1.5, zorder=3)
        pts = r[r.setting == s]["pass"] * 100
        ax.scatter(pts, [i] * len(pts), s=14, color="white", edgecolor="k", zorder=4, linewidth=0.8)
        ax.text(max(hi, m) + 1.2, i, f"{m:.1f}  [{lo:.1f}–{hi:.1f}] n={int(tab.loc[s, 'runs'])}", va="center", fontsize=8.5,
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85), zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[s] for s in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("pass (%)  막대=반복 평균, 검은 선=최소–최대, 흰 점=개별 실행, n=반복 수")
    ax.set_title(title, fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=BLUE_D), Patch(color=RED_D)]
    labels += ["RAG (진하게: 기준·핵심)", "Agent (진하게: 핵심)"]
    ax.legend(handles, labels, loc="lower right", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("답변 단계 pass % (설정별 반복 평균과 범위)", fontsize=14, y=0.995)
fig.tight_layout()
save(fig, "fig_answer_pass.png")

# ---------- 3. 비용 대비 품질 ----------
OFF = {"base": {"B0": (-30, 6), "E4-a": (8, -8), "E3-a": (4, -14), "E8a": (-14, 8), "E3-b": (-36, -12), "C0": (9, -3),
                "A1": (-24, 8), "A2": (8, -12), "A4": (8, 4), "E7": (8, 4), "E7a": (-30, 8), "E6-a": (-30, -4),
                "E6-b": (8, -12), "E8b": (-8, 8)},
       "ext": {"B0": (8, -12), "E7a": (8, 4), "E4-a": (8, -2), "E4-b": (-32, -4), "E3-a": (8, -10), "E6-a": (8, -3),
               "E6-b": (-8, 8), "E8a": (-34, -12), "E7": (8, 6), "C0": (-32, 6), "A4": (9, 6), "A2": (8, -12),
               "A1": (8, 6), "E3-b": (8, -4), "E8b": (-8, 8)}}
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
for ax, (sn, title) in zip(axes, SETS):
    r = runs[runs.set_name == sn]
    g = r.groupby("setting").agg(x=("input_tokens_mean", "mean"), y=("pass", "mean"), calls=("llm_calls_mean", "mean"),
                                 stage=("stage", "first"))
    for s, row in g.iterrows():
        base = s == "B0"
        ax.scatter(row.x, row.y * 100, s=row.calls * 70 + 20, color=RED if row.stage == "agent" else BLUE, alpha=0.8,
                   edgecolor="k" if base else "white", linewidth=2.2 if base else 0.8, zorder=3)
        ax.annotate(s, (row.x, row.y * 100), xytext=OFF[sn].get(s, (6, 4)), textcoords="offset points", fontsize=8.5,
                    fontweight="bold" if (s in KEY or base) else "normal")
    ax.set_xlabel("입력 토큰 / 문항")
    ax.set_ylabel("pass (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12, y=0.12)
h = [Line2D([], [], marker="o", ls="", color=BLUE, label="RAG"), Line2D([], [], marker="o", ls="", color=RED, label="Agent"),
     Line2D([], [], marker="o", ls="", color="white", markeredgecolor="k", markeredgewidth=2, label="B0 기준 (검은 테두리)"),
     Line2D([], [], marker="o", ls="", color="gray", markersize=5, label="점 크기 = LLM 호출 수/문항")]
axes[1].legend(handles=h, loc="lower right", fontsize=8)
fig.suptitle("비용 대비 품질 (입력 토큰 vs pass, 설정별 반복 평균)", fontsize=14)
fig.tight_layout()
save(fig, "fig_cost_quality.png")

# ---------- 4. 기대 행동 히트맵 ----------
gr = D.answer_groups(SRC)
gr = gr[(gr.group_type == "expected_group") & (gr.set_name == "base")]
rows = ["B0", "E3-b", "E7", "E7a", "C0", "A2", "A4"]
cols = ["answer", "correct", "ask", "abstain", "refuse"]
mat = np.array([[gr[(gr.setting == s) & (gr.group == c)]["pass"].mean() * 100 for c in cols] for s in rows])
ns = {c: int(gr[gr.group == c].n.iloc[0]) for c in cols}
fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(5))
ax.set_xticklabels([f"{D.BEHAVIOR_KO[c]} ({ns[c]})" for c in cols], fontsize=11)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([SHORT[s] for s in rows], fontsize=10)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel("기대 행동 (괄호 = 문항 수)", fontsize=10)
ax.set_ylabel("설정")
for i in range(len(rows)):
    for j in range(5):
        v = mat[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=11, color="white" if v > 60 else "black")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("pass (%)")
fig.suptitle("기대 행동별 pass % (기본 세트, 반복 평균)", fontsize=14, y=0.995)
fig.tight_layout()
save(fig, "fig_behavior_heatmap.png")

if "--print" in sys.argv:
    print(rows)
    print(np.round(mat, 1))

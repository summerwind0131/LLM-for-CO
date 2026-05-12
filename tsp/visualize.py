# =============================================================================
#  模块九：可视化
# =============================================================================

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import (
    SEEDS,
    STRATEGIES,
    USE_TWO_OPT,
    OUTPUT_DIR,
    RUN_ID,
)


def plot_results(
    all_histories: dict,
    final_results: dict,
    instance_name: str,
    llm_logs:      list = None,
):
    """
    双图布局：
    左图 — 多种子均值收敛曲线 + 标准差阴影 + LLM触发时间线
    右图 — 最终解质量箱线图
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"SC-LLM-OS Ablation Study  |  {instance_name}  "
        f"|  Seeds={SEEDS}  |  2-opt={'ON' if USE_TWO_OPT else 'OFF'}",
        fontsize=12, fontweight="bold",
    )

    style_map = {
        "baseline":         ("gray",       "--",  "Baseline (Uniform)"),
        "traditional_alns": ("darkorange", "-.",  "Traditional ALNS"),
        "sc_llm_os":        ("royalblue",  "-",   "SC-LLM-OS (Ours)"),
    }

    # ── 左图：均值收敛曲线 ──────────────────────────────────────────────────
    ax = axes[0]
    for strategy in STRATEGIES:
        hists = all_histories.get(strategy, [])
        if not hists:
            continue
        arr        = np.array(hists)
        mean_curve = arr.mean(axis=0)
        std_curve  = arr.std(axis=0)
        color, ls, label = style_map[strategy]
        x        = np.arange(len(mean_curve))
        best_val = float(np.min(arr))
        ax.plot(mean_curve,
                label=f"{label}  (best={best_val:.1f})",
                color=color, linestyle=ls, linewidth=2.2, zorder=3)
        ax.fill_between(x,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.12, color=color, zorder=2)

    # LLM 触发时间线标注
    if llm_logs:
        mode_color = {
            "exploit":          "#27ae60",
            "explore_topology": "#e67e22",
            "explore_global":   "#e74c3c",
        }
        plotted_modes = set()
        for entry in llm_logs:
            is_fb = entry.get("is_fallback", False)
            mc    = mode_color.get(entry["mode"], "purple")
            lbl   = None
            if entry["mode"] not in plotted_modes and not is_fb:
                lbl = f"LLM: {entry['mode']}"
                plotted_modes.add(entry["mode"])
            ax.axvline(x        = entry["iteration"],
                       color    = mc,
                       alpha    = 0.30,
                       linewidth= 1.4 if is_fb else 0.8,
                       linestyle= ":" if is_fb else "-",
                       label    = lbl,
                       zorder   = 1)

    ax.set_title("Mean Convergence Curve ± Std Dev", fontsize=11)
    ax.set_xlabel("Iterations", fontsize=10)
    ax.set_ylabel("Best Distance Found", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # ── 右图：箱线图 ────────────────────────────────────────────────────────
    ax2        = axes[1]
    box_data   = [final_results.get(s, []) for s in STRATEGIES]
    box_labels = ["Baseline", "Trad-ALNS", "SC-LLM-OS"]
    colors_box = ["gray", "darkorange", "royalblue"]

    bp = ax2.boxplot(
        box_data,
        tick_labels  = box_labels,
        patch_artist = True,
        widths       = 0.5,
        medianprops  = dict(color="white", linewidth=2.0),
        whiskerprops = dict(linewidth=1.4),
        capprops     = dict(linewidth=1.4),
        flierprops   = dict(marker="o", markersize=5, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, (strategy, color) in enumerate(zip(STRATEGIES, colors_box), start=1):
        data = final_results.get(strategy, [])
        if data:
            mv = float(np.mean(data))
            ax2.text(i, mv, f"  μ={mv:.1f}",
                     fontsize=8, color=color,
                     va="center", fontweight="bold")

    dirty = final_results.get("sc_llm_os_dirty_seeds", [])
    title_suffix = (f"\n⚠️ SC-LLM-OS: {len(dirty)} dirty seed(s)"
                    if dirty else "")
    ax2.set_title(f"Final Solution Quality{title_suffix}", fontsize=11)
    ax2.set_ylabel("Best Distance Found", fontsize=10)
    ax2.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"results_{instance_name}_{RUN_ID}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n   📊 收敛图已保存: {save_path}")
    plt.close()


def plot_llm_decision_timeline(
    llm_logs:      list,
    instance_name: str,
    seed:          int,
):
    """
    LLM 决策时间线图（答辩展示用）。
    正常决策用彩色符号标注，Fallback 用红色 × 标注。
    """
    if not llm_logs:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    mode_color  = {
        "exploit":          "#27ae60",
        "explore_topology": "#e67e22",
        "explore_global":   "#c0392b",
    }
    mode_marker = {
        "exploit":          "^",
        "explore_topology": "s",
        "explore_global":   "D",
    }

    iterations = [e["iteration"]     for e in llm_logs]
    best_dists = [e["best_distance"]  for e in llm_logs]

    ax.plot(iterations, best_dists,
            color="royalblue", linewidth=1.5, zorder=2,
            label="Best Distance at Trigger")

    for entry in llm_logs:
        it     = entry["iteration"]
        bd     = entry["best_distance"]
        mode   = entry["mode"]
        reason = entry.get("reasoning", "")
        is_fb  = entry.get("is_fallback", False)

        c  = mode_color.get(mode, "gray")
        mk = "x" if is_fb else mode_marker.get(mode, "o")
        sz = 130 if is_fb else 80
        ax.scatter(it, bd, color="red" if is_fb else c,
                   marker=mk, s=sz, zorder=3,
                   linewidths=2.5 if is_fb else 1.0)

        text = (f"[FB]\n{mode}" if is_fb
                else (f"{mode}\n{reason[:26]}…"
                      if len(reason) > 26 else f"{mode}\n{reason}"))
        ax.annotate(
            text,
            xy         = (it, bd),
            xytext     = (0, 16),
            textcoords = "offset points",
            fontsize   = 6,
            color      = "red" if is_fb else c,
            ha         = "center",
            arrowprops = dict(arrowstyle="-",
                              color="red" if is_fb else c,
                              lw=0.6),
        )

    legend_patches = [
        mpatches.Patch(color=mode_color["exploit"],
                       label="exploit"),
        mpatches.Patch(color=mode_color["explore_topology"],
                       label="explore_topology"),
        mpatches.Patch(color=mode_color["explore_global"],
                       label="explore_global"),
        mpatches.Patch(color="red",
                       label="FALLBACK (dirty)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right")
    ax.set_title(
        f"LLM Decision Timeline  |  {instance_name}  |  Seed={seed}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Iteration",                  fontsize=10)
    ax.set_ylabel("Best Distance at Trigger",   fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR, f"llm_timeline_{instance_name}_seed{seed}_{RUN_ID}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   🧠 LLM时间线图已保存: {save_path}")
    plt.close()

"""Publication graphics for the n=28 seed study (runs on thunderlane, venv).

Reads seed_stats.json (produced by analyze_seeds.py) and writes four figures
to results/figures/ as SVG + 200-dpi PNG:
  1. slopegraph_valnll   - 28 paired seeds, baseline -> dynamic edges
  2. paired_diffs        - per-seed deltas + 95% CI (val NLL, test ANLL)
  3. evidence_trajectory - cumulative effect estimate as seeds accumulate
  4. ade_equivalence     - the point-accuracy null (ADE / FDE deltas)
"""
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# chronological order: pilot, extension, confirmation campaign
SEEDS = [1234, 1, 42, 7, 13, 99, 2024, 3407,
         2, 3, 5, 11, 17, 23, 31, 37, 55, 77, 101, 123, 222,
         314, 555, 777, 888, 1000, 2718, 9999]
BLUE, GREEN, ORANGE, INK, MUT = "#2a78d6", "#008300", "#eb6834", "#222222", "#555555"
OUT = "results/figures"

matplotlib.rcParams.update({
    "svg.fonttype": "path", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
})


def style(ax, xgrid=False):
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    if xgrid:
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors=MUT, labelsize=8)


def save(fig, name):
    for ext, kw in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight",
                    facecolor="white", **kw)
    plt.close(fig)
    print(f"[fig] {name}")


def series(paired, key, which):
    return np.array([paired[key][f"per_seed_{which}"][str(s)] for s in SEEDS])


def ci95(d):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return d.mean() - h, d.mean() + h


def main():
    os.makedirs(OUT, exist_ok=True)
    paired = json.load(open("seed_stats.json"))["paired"]["dynedge_vs_baseline"]
    base = series(paired, "best_val_nll", "a")
    dyn = series(paired, "best_val_nll", "b")
    d_nll = dyn - base

    # 1 -- slopegraph ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for y0, y1 in zip(base, dyn):
        ax.plot([0, 1], [y0, y1], color="#c8c8c8", lw=0.8, zorder=1)
    ax.plot(np.zeros_like(base), base, "o", ms=5, color=BLUE,
            markeredgecolor="white", markeredgewidth=0.9, zorder=2)
    ax.plot(np.ones_like(dyn), dyn, "o", ms=5, color=GREEN,
            markeredgecolor="white", markeredgewidth=0.9, zorder=2)
    ax.plot([0, 1], [base.mean(), dyn.mean()], color=INK, lw=2.4, zorder=3)
    ax.annotate(f"mean {base.mean():.2f}", (0, base.mean()),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9, color=INK)
    ax.annotate(f"mean {dyn.mean():.2f}", (1, dyn.mean()),
                xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9, color=INK)
    wins = int((d_nll < 0).sum())
    r = paired["best_val_nll"]
    ax.set_title(f"Best validation NLL, {len(SEEDS)} paired seeds — "
                 f"dynamic edges better in {wins}/{len(SEEDS)}", loc="left")
    ax.set_xticks([0, 1], ["baseline", "dynamic edges"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("best validation NLL (nats, lower = better)")
    style(ax)
    save(fig, "slopegraph_valnll")

    # 2 -- paired differences with CI -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, key, label, unit in (
            (axes[0], "best_val_nll", "best validation NLL", "nats"),
            (axes[1], "test_anll", "test ANLL", "nats")):
        d = series(paired, key, "b") - series(paired, key, "a")
        t, p = stats.ttest_rel(series(paired, key, "b"),
                               series(paired, key, "a"))
        order = np.argsort(d)
        x = np.arange(len(d))
        lo, hi = ci95(d)
        ax.axhspan(lo, hi, color="#e9eef7", zorder=0)
        ax.axhline(d.mean(), color=INK, lw=1.2, zorder=1)
        ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
        colors = [BLUE if v < 0 else ORANGE for v in d[order]]
        ax.scatter(x, d[order], c=colors, s=26, edgecolor="white",
                   linewidth=0.8, zorder=2)
        ax.set_title(f"{label}: paired Δ (dyn − base)\n"
                     f"mean {d.mean():+.2f} {unit}, 95% CI [{lo:+.2f}, {hi:+.2f}], "
                     f"p={p:.3f}", loc="left", fontsize=9)
        ax.set_xlabel("seeds (sorted by Δ)")
        ax.set_ylabel(f"Δ ({unit}); below 0 = improvement")
        style(ax)
    fig.tight_layout(w_pad=2.5)
    save(fig, "paired_diffs")

    # 3 -- evidence trajectory --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, key, label in ((axes[0], "best_val_nll", "best validation NLL"),
                           (axes[1], "test_anll", "test ANLL")):
        d = series(paired, key, "b") - series(paired, key, "a")
        ns, means, los, his, ps = [], [], [], [], {}
        for n in range(3, len(d) + 1):
            dd = d[:n]
            ns.append(n)
            means.append(dd.mean())
            lo, hi = ci95(dd)
            los.append(lo)
            his.append(hi)
            if n in (3, 8, len(d)):
                ps[n] = stats.ttest_rel(series(paired, key, "b")[:n],
                                        series(paired, key, "a")[:n]).pvalue
        ax.fill_between(ns, los, his, color="#e9eef7", zorder=0,
                        label="95% CI")
        ax.plot(ns, means, color=BLUE, lw=1.8, zorder=2,
                label="cumulative mean Δ")
        ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
        for n, p in ps.items():
            i = n - 3
            ax.annotate(f"n={n}\np={p:.3f}", (n, means[i]),
                        xytext=(0, 14), textcoords="offset points",
                        ha="center", fontsize=7.5, color=MUT)
        ax.set_title(f"{label}: effect estimate vs. seeds", loc="left",
                     fontsize=9)
        ax.set_xlabel("number of paired seeds (chronological)")
        ax.set_ylabel("Δ (nats); below 0 = improvement")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        style(ax)
    fig.tight_layout(w_pad=2.5)
    save(fig, "evidence_trajectory")

    # 4 -- point-accuracy equivalence -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, key, label, band in ((axes[0], "test_ade", "test ADE", 0.05),
                                 (axes[1], "test_fde", "test FDE", 0.10)):
        d = series(paired, key, "b") - series(paired, key, "a")
        t, p = stats.ttest_rel(series(paired, key, "b"),
                               series(paired, key, "a"))
        order = np.argsort(d)
        x = np.arange(len(d))
        ax.axhspan(-band, band, color="#f2f2ee", zorder=0)
        ax.annotate(f"±{band:.2f} m band", (0.3, band), fontsize=7.5,
                    color=MUT, va="bottom")
        lo, hi = ci95(d)
        ax.axhspan(lo, hi, color="#e9eef7", zorder=1)
        ax.axhline(d.mean(), color=INK, lw=1.2, zorder=2)
        ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=2)
        ax.scatter(x, d[order], color=MUT, s=24, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.set_title(f"{label}: paired Δ (dyn − base)\n"
                     f"mean {d.mean():+.3f} m, 95% CI [{lo:+.3f}, {hi:+.3f}], "
                     f"p={p:.2f}", loc="left", fontsize=9)
        ax.set_xlabel("seeds (sorted by Δ)")
        ax.set_ylabel("Δ (m)")
        style(ax)
    fig.tight_layout(w_pad=2.5)
    save(fig, "ade_equivalence")


if __name__ == "__main__":
    main()

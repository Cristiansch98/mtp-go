"""Build the MTP-GO improvement paper PDF.

Runs on thunderlane inside the mtp-go venv:
  1. renders every equation delimited by &#10218; ... &#10219; in paper.html to a
     baseline-aligned SVG (matplotlib mathtext, STIX fonts),
  2. generates fig1.svg from the training-run metric CSVs in ../logs/,
  3. renders the final PDF with WeasyPrint and reports the page count.
"""
import base64
import glob
import html
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser

HERE = os.path.dirname(os.path.abspath(__file__))
EQ_DIR = os.path.join(HERE, "eqs")
LOG_ROOT = os.path.join(HERE, "..", "logs")

matplotlib.rcParams.update({
    "mathtext.fontset": "stix",
    "svg.fonttype": "path",
    "font.size": 9.3,
})

BODY_PT = 9.3
PAD_PT = 0.6


def render_equation(tex, out_path, fontsize=BODY_PT):
    """Render one mathtext string to SVG; return (width, height, depth) in pt."""
    tex = tex.replace(r"\|", r"\Vert ")
    prop = FontProperties(size=fontsize)
    parser = MathTextParser("path")
    parsed = parser.parse(f"${tex}$", dpi=72, prop=prop)
    width, height, depth = parsed.width, parsed.height, parsed.depth
    w, h = width + 2 * PAD_PT, height + 2 * PAD_PT
    fig = plt.figure(figsize=(w / 72.0, h / 72.0))
    fig.patch.set_alpha(0.0)
    fig.text(PAD_PT / w, (depth + PAD_PT) / h, f"${tex}$",
             fontsize=fontsize, va="baseline", ha="left")
    fig.savefig(out_path, format="svg", transparent=True)
    plt.close(fig)
    return w, h, depth + PAD_PT


def build_equations(src):
    os.makedirs(EQ_DIR, exist_ok=True)
    pattern = re.compile(r"&#10218;(.*?)&#10219;", re.S)
    cache = {}

    def repl(m):
        tex = html.unescape(m.group(1)).strip()
        if tex not in cache:
            name = f"eq_{len(cache):03d}.svg"
            path = os.path.join(EQ_DIR, name)
            w, h, d = render_equation(tex, path)
            cache[tex] = (name, w, h, d)
        name, w, h, d = cache[tex]
        return (f'<img class="mi" src="eqs/{name}" '
                f'style="height:{h:.2f}pt;vertical-align:{-d:.2f}pt"/>')

    out = pattern.sub(repl, src)
    print(f"[eqs] rendered {len(cache)} unique equations")
    return out


# ---------------------------------------------------------------- figure ----
COLORS = {  # validated categorical palette, slots 1-4 (light mode)
    "baseline": "#2a78d6",
    "dynedge": "#008300",
    "full-topo": "#e87ba4",
    "full+dyn": "#eda100",
}
RUNS_A = {  # panel (a): seed-1234 histories
    "baseline": "SecondOrderNeuralODE64G1InD_15-07*",
    "dynedge": "SecondOrderNeuralODE64G1DEInD_DYNEDGE*",
    "full-topo": "SecondOrderNeuralODE64G1FEInD_FULL_*",
    "full+dyn": "SecondOrderNeuralODE64G1DEFEInD_FULLDYN*",
}
SEEDS = ["1234", "1", "42", "7", "13", "99", "2024", "3407"]
RUNS_B = {  # panel (b): best val NLL per seed; trailing _ disambiguates S1/S13
    "baseline": {s: (f"*_BASE_S{s}_*" if s != "1234"
                     else "SecondOrderNeuralODE64G1InD_15-07*") for s in SEEDS},
    "dynedge": {s: (f"*_DYN_S{s}_*" if s != "1234"
                    else "SecondOrderNeuralODE64G1DEInD_DYNEDGE*") for s in SEEDS},
}


def load_val_nll(pat):
    """Return (epochs, val_nll) from the metrics.csv under a log-dir glob."""
    import pandas as pd
    hits = sorted(glob.glob(os.path.join(LOG_ROOT, pat, "**", "metrics.csv"),
                            recursive=True))
    if not hits:
        raise FileNotFoundError(f"no metrics.csv under {pat}")
    df = pd.read_csv(hits[-1])
    df = df[df["val_nll"].notna()]
    return df["epoch"].to_numpy(), df["val_nll"].to_numpy()


def style_axes(ax):
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=6.8, width=0.6, length=2.5)


def build_figure(out_path):
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.25, 4.1), height_ratios=[1.15, 1.0],
        constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.12)

    # (a) validation NLL curves, post objective-schedule transition
    for name, pat in RUNS_A.items():
        ep, nll = load_val_nll(pat)
        m = ep >= 30
        ax_a.plot(ep[m], nll[m], color=COLORS[name], linewidth=1.2, label=name)
    ax_a.set_ylim(-5.5, 12)
    ax_a.set_xlim(30, 202)
    ax_a.set_xlabel("epoch", fontsize=7.2, color="#222222")
    ax_a.set_ylabel("validation NLL (nats)", fontsize=7.2, color="#222222")
    ax_a.legend(frameon=False, fontsize=6.6, loc="upper right",
                handlelength=1.4, labelspacing=0.3)
    ax_a.set_title("(a)  Validation NLL, seed 1234", fontsize=7.5,
                   color="#222222", loc="left", pad=3)
    style_axes(ax_a)

    # (b) best val NLL per seed, paired over all 8 seeds
    for seed in SEEDS:
        ys = []
        for variant in ("baseline", "dynedge"):
            _, nll = load_val_nll(RUNS_B[variant][seed])
            ys.append(nll.min())
        ax_b.plot([0, 1], ys, color="#bbbbbb", linewidth=0.9, zorder=1)
        for xi, variant in enumerate(("baseline", "dynedge")):
            ax_b.plot(xi, ys[xi], "o", ms=5, color=COLORS[variant],
                      markeredgecolor="white", markeredgewidth=1.0, zorder=2)
    ax_b.set_xlim(-0.35, 1.35)
    ax_b.set_xticks([0, 1], ["baseline", "dynamic edges"])
    ax_b.set_ylabel("best validation NLL (nats)", fontsize=7.2,
                    color="#222222")
    ax_b.set_title("(b)  Best validation NLL by seed", fontsize=7.5,
                   color="#222222", loc="left", pad=3)
    style_axes(ax_b)

    fig.savefig(out_path, format="svg", transparent=True)
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def main():
    src = open(os.path.join(HERE, "paper.html")).read()
    if "--skip-fig" not in sys.argv:
        build_figure(os.path.join(HERE, "fig1.svg"))
    built = build_equations(src)
    built_path = os.path.join(HERE, "paper_built.html")
    open(built_path, "w").write(built)

    from weasyprint import HTML
    doc = HTML(filename=built_path, base_url=HERE).render()
    out_pdf = os.path.join(HERE, "paper.pdf")
    doc.write_pdf(out_pdf)
    print(f"[pdf] {out_pdf}: {len(doc.pages)} pages")


if __name__ == "__main__":
    main()

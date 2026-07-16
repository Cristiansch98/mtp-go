"""Multi-seed statistical analysis: baseline vs dynamic edges (n=8),
full-topology variants (n=3). Runs on thunderlane in the mtp-go venv,
from the repo root (JSONs in cwd, histories under logs/).

Outputs a readable report and a machine-readable seed_stats.json.
"""
import glob
import json
import math
import os

import numpy as np
from scipy import stats

SEEDS8 = [1234, 1, 42, 7, 13, 99, 2024, 3407]
SEEDS3 = [1234, 1, 42]

# per-variant: (json name, log-dir glob) keyed by seed; 1234 = original runs
def named(prefix, tag, seed, orig_json, orig_glob):
    if seed == 1234:
        return orig_json, orig_glob
    return (f"{prefix}_{tag}_S{seed}.json", f"logs/*_{tag}_S{seed}_*")

VARIANTS = {
    "baseline": (SEEDS8, lambda s: named(
        "SecondOrderNeuralODE64G1InD", "BASE", s,
        "SecondOrderNeuralODE64G1InD.json",
        "logs/SecondOrderNeuralODE64G1InD_15-07*")),
    "dynedge": (SEEDS8, lambda s: named(
        "SecondOrderNeuralODE64G1DEInD", "DYN", s,
        "SecondOrderNeuralODE64G1DEInD_DYNEDGE.json",
        "logs/*_DYNEDGE_*")),
    "full": (SEEDS3, lambda s: named(
        "SecondOrderNeuralODE64G1FEInD", "FULL", s,
        "SecondOrderNeuralODE64G1FEInD_FULL.json",
        "logs/*FEInD_FULL_1*")),
    "fulldyn": (SEEDS3, lambda s: named(
        "SecondOrderNeuralODE64G1DEFEInD", "FULLDYN", s,
        "SecondOrderNeuralODE64G1DEFEInD_FULLDYN.json",
        "logs/*_FULLDYN_1*")),
}

TEST_KEYS = ["test_ade", "test_fde", "test_anll", "mr"]


def best_val(pattern):
    import pandas as pd
    hits = sorted(glob.glob(os.path.join(pattern, "**", "metrics.csv"),
                            recursive=True))
    assert hits, f"no metrics.csv for {pattern}"
    df = pd.read_csv(hits[-1])
    df = df[df["val_nll"].notna()]
    return {"best_val_nll": float(df["val_nll"].min()),
            "best_val_ade": float(df["val_ade"].min()),
            "best_val_fde": float(df["val_fde"].min())}


def collect(variant):
    seeds, namer = VARIANTS[variant]
    rows = {}
    for s in seeds:
        jname, lglob = namer(s)
        j = json.load(open(jname))
        if isinstance(j, list):
            j = j[0]
        row = {k: j.get(k) for k in TEST_KEYS}
        row.update(best_val(lglob))
        rows[s] = row
    return rows


def paired(a, b, key):
    """a, b: dict seed->row over identical seeds. Returns stats for b-a."""
    seeds = list(a.keys())
    xa = np.array([a[s][key] for s in seeds])
    xb = np.array([b[s][key] for s in seeds])
    d = xb - xa
    n = len(d)
    res = {
        "n": n,
        "mean_a": float(xa.mean()), "std_a": float(xa.std(ddof=1)),
        "mean_b": float(xb.mean()), "std_b": float(xb.std(ddof=1)),
        "mean_diff": float(d.mean()),
        "wins_b": int((d < 0).sum()),  # all metrics: lower is better
        "per_seed_a": {str(s): float(a[s][key]) for s in seeds},
        "per_seed_b": {str(s): float(b[s][key]) for s in seeds},
    }
    if n >= 3 and d.std(ddof=1) > 0:
        t, p = stats.ttest_rel(xb, xa)
        res["t"] = float(t)
        res["p_t"] = float(p)
        res["cohens_d"] = float(d.mean() / d.std(ddof=1))
        try:
            w, pw = stats.wilcoxon(xb, xa, mode="exact")
            res["p_wilcoxon"] = float(pw)
        except Exception:
            pass
    return res


def main():
    data = {v: collect(v) for v in VARIANTS}
    keys = TEST_KEYS + ["best_val_nll", "best_val_ade", "best_val_fde"]
    out = {"per_variant": data, "paired": {}}

    for name, (a, b) in {"dynedge_vs_baseline": ("baseline", "dynedge"),
                         "fulldyn_vs_full": ("full", "fulldyn")}.items():
        out["paired"][name] = {k: paired(data[a], data[b], k) for k in keys}

    json.dump(out, open("seed_stats.json", "w"), indent=1)

    for name, block in out["paired"].items():
        print(f"\n==== {name} ====")
        for k, r in block.items():
            line = (f"{k:14s} A {r['mean_a']:8.3f}±{r['std_a']:.3f}  "
                    f"B {r['mean_b']:8.3f}±{r['std_b']:.3f}  "
                    f"diff {r['mean_diff']:+.3f}  B-wins {r['wins_b']}/{r['n']}")
            if "t" in r:
                line += (f"  t={r['t']:+.2f} p={r['p_t']:.4f}"
                         f" d={r['cohens_d']:+.2f}")
                if "p_wilcoxon" in r:
                    line += f" pW={r['p_wilcoxon']:.4f}"
            print(line)


if __name__ == "__main__":
    main()

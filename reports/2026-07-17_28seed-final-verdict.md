# Final Verdict at n=28: The Dynamic-Edges Calibration Effect Was Seed Noise

**Date:** 2026-07-17 · **Supersedes** the 3-seed and 8-seed reports · **Data:** `results/seed_stats.json` · **Graphics:** `results/figures/` · **Runs:** 56 paired (28 seeds × baseline/dynedge) + 6 topology-ablation + 3 secondary = 65 total

## Design

Pre-planned confirmation: the 8-seed interim showed d≈0.54, so 20 more seeds
(n=28) were fixed **in advance** for 80% power at α=0.05 — one test at the
planned n, no sequential peeking. 40 new runs (~11 h, 4 streams, zero errors).

## Result: null

| Metric | Baseline (n=28) | Dynedge (n=28) | Δ (95% CI) | p (t / Wilcoxon) |
|---|---|---|---|---|
| best val NLL | −0.88 ± 2.70 | −0.65 ± 3.59 | +0.23 [−0.97, +1.44] | 0.70 / 0.87 |
| test ANLL | −0.088 ± 0.106 | −0.079 ± 0.142 | +0.01 [−0.04, +0.06] | 0.70 / 0.96 |
| test ADE (m) | 1.020 ± 0.042 | 1.030 ± 0.049 | +0.010 [−0.004, +0.023] | 0.15 / 0.17 |
| test FDE (m) | 2.889 ± 0.112 | 2.918 ± 0.122 | +0.029 [−0.002, +0.060] | 0.07 / 0.13 |

Wins: 14/28 on val NLL (coin flip). Evidence trajectory: −2.41 nats (n=3, 3/3)
→ −1.11 (n=8, 6/8, p=0.17) → **+0.23 (n=28, p=0.70)**. Textbook winner's curse:
baseline val NLL spans −4.9…+7.6 across seeds (some seeds never calibrate);
σ≈2.7–3.6 nats makes a 3-seed −2.4-nat "effect" unremarkable noise.

## What survives (and is now strongly supported)

1. **The leak-free decoder** (`--full-edges`): causal, point-for-point
   equivalent to the oracle-fed original (n=3 seed-matched: +0.013 m ADE).
   MTP-GO's published accuracy stands on causal footing. **Adopt.**
2. **The accuracy equivalence null** for dynamic edges: ΔADE CI [−0.004, +0.023] m.
3. **The noise-band quantification** for this model family: ±0.04 m ADE,
   ±0.1 nats test ANLL, ±2.7–3.6 nats best val NLL across seeds — wider than
   many published SOTA margins.

## Recommendation changes

- `--dynamic-edges`: **no longer recommended** as a default (no demonstrated
  benefit, ~25% training cost, accuracy CI leans negative; best val FDE
  nominally significant against, p=0.03, pre-correction).
- Paper retitled and rewritten around the two durable results: the deployable
  leak-free decoder and the 28-seed cautionary study (Fig. 2 = evidence
  trajectory). H1 rejected, H2 unproven, H3 confirmed.

## Graphics

- `slopegraph_valnll` — 28 paired seeds, flat means (−0.88 vs −0.65)
- `paired_diffs` — per-seed Δ with 95% CI (val NLL, test ANLL)
- `evidence_trajectory` — the 3→8→28 regression-to-null curve
- `ade_equivalence` — point-accuracy deltas inside ±0.05/±0.10 m bands

# Dynamic Decoder Edges for MTP-GO: Multi-Seed Evaluation Report

**Date:** 2026-07-16 · **Dataset:** inD (recordings 18–29, 7,820 samples, 80/10/10 split) · **Hardware:** RTX 5090, torch 2.8.0+cu128 · **Code:** [Cristiansch98/mtp-go](https://github.com/Cristiansch98/mtp-go), feature commit `7ba31fe`

## 1. Background and hypothesis

MTP-GO (Westny et al., IEEE T-IV 2023) predicts multi-agent trajectories with a GRU-GNN
encoder–decoder and neural-ODE motion models. A code-level analysis of the decoder
established two facts sharper than the known "frozen graph" caveat:

1. The decoder consumes the **ground-truth future interaction topology**
   (`tar_edge_index[di]`) at every one of the 25 rollout steps — during training *and*
   evaluation (a future-information leak, albeit consistent across phases).
2. The decoder receives **no edge features at all**: its GNN cell is constructed with
   `edge_dim=None`, and the precomputed future distances (`tar_edge_features`) are dead
   data, never loaded by any model code. Only the encoder sees inter-agent distances,
   through a Gaussian kernel `exp(-(d/bw)²)` with learnable bandwidth.

**Change under test (`--dynamic-edges`).** At each decoder step, inter-agent Euclidean
distances are recomputed from the π-weighted mean of the fed-back per-mixture positions
(the model's own detached predictions; ground truth under teacher forcing) and passed to
the decoder GNN cell (`edge_dim=1`) through the same learnable Gaussian kernel as the
encoder. Positions come from step *di−1* by design — that is all the model knows when
predicting step *di*; using `tar_edge_features[di]` would reintroduce the leak. Topology
is left unchanged (graphs are complete over present agents, so interaction *structure* is
carried by the distances). Gradients flow only through the GNN/bandwidth path, mirroring
the existing detached state feedback. Training-throughput cost: ≈25%.

**Hypothesis:** giving the decoder a self-consistent, evolving picture of inter-agent
proximity improves prediction quality over the static/no-feature decoder.

## 2. Experimental setup

Both variants use the paper-default configuration (SecondOrderNeuralODE motion model,
hidden 64, one `natt` GNN layer, M=8 mixtures, rk4, batch 128, 200 epochs, annealed
teacher forcing 0.2→0). **Three seeds per variant** (1234, 1, 42); identical data splits
(fixed at preprocessing). Metrics follow the paper: most-likely-component ADE/FDE (m),
average NLL, and miss rate (2 m at 4 s) on the 920-sample test split.

## 3. Results

**Test set, mean ± std over 3 seeds (per-seed values in brackets):**

| Metric | Baseline | Dynamic edges | Paired Δ (per seed) |
|---|---|---|---|
| ADE (m) | 1.013 ± 0.034 [1.006, 0.983, 1.049] | 1.018 ± 0.038 [1.036, 0.975, 1.043] | +0.005 (t=0.4, n.s.) |
| FDE (m) | 2.852 ± 0.118 | 2.897 ± 0.118 | +0.045 (n.s.) |
| **ANLL** | −0.095 ± 0.018 [−0.108, −0.074, −0.102] | **−0.173 ± 0.064** [−0.197, −0.222, −0.101] | **−0.078** (t=−1.8) |
| Miss rate | 0.418 ± 0.030 | 0.405 ± 0.027 | −0.013 (n.s.) |

**Best validation NLL** (calibration during training):

| | Baseline | Dynamic edges | Paired Δ |
|---|---|---|---|
| best val NLL | −0.61 ± 0.59 [−1.27, −0.12, −0.43] | **−3.02 ± 1.61** [−3.63, −4.22, −1.19] | **−2.41 nats** (t=−2.5, better in 3/3 seeds) |
| best val ADE | 1.003 ± 0.032 | 0.986 ± 0.028 | −0.017 (n.s.) |

**Findings.**
- **Calibration improves consistently.** Dynamic edges win on validation NLL in **3 of 3
  seeds** with a large average margin (≈2.4 nats), and on test ANLL in 2 of 3 seeds (one
  tie). With n=3 the paired t-tests do not reach the conventional 0.05 threshold
  (p≈0.13–0.21), but the effect is direction-consistent and an order of magnitude larger
  than the displacement differences.
- **Point accuracy is unchanged.** ADE/FDE/miss-rate differences (±0.005–0.05) are well
  inside seed noise (baseline ADE alone spans 0.983–1.049 across seeds). Single-seed
  comparisons of this model family are unreliable below ≈0.07 m ADE.
- **Interpretation.** The evolving inter-agent distances feed the GNN that generates the
  mixture process noise, so the information sharpens the predicted *covariances and mode
  weights* rather than the point estimate — visible qualitatively as scene-adaptive
  uncertainty ellipses (`results/trajectories_baseline_vs_dynedge.png`).

**Secondary ablations** (single seed, context): the paper's M=8 mixtures proved locally
optimal (M=4: ADE 1.096; M=16: 1.067; both worse than 1.006), and the integrator is
load-bearing — replacing rk4 with Euler-forward degraded ADE by +51% (1.516) with val
ADE diverging after ~epoch 50. Full curves: `results/comparison_inD.png`.

## 4. Conclusions and recommendations

1. **Adopt `--dynamic-edges` when probabilistic quality matters** (planning under
   uncertainty, risk assessment): a consistent, large calibration gain at ≈25% training
   overhead and no accuracy cost. For pure point-accuracy benchmarks it is neutral.
2. The "frozen graph" critique is confirmed but nuanced: on inD (low speeds, dense
   interactions) its cost is in uncertainty quality, not displacement. Highway data
   (highD) with ~150 m of travel per horizon may respond differently — untested.
3. **Remaining leak:** future *presence* topology still comes from ground truth. A fully
   deployable decoder needs presence-agnostic graphs (e.g. the preprocessed but unused
   `full_edge_idx`) — the natural next experiment.
4. With n=3 seeds, NLL variance is high (seed 42 is an outlier in both variants); 5–10
   seeds would be needed to claim significance formally.

**Reproduce:** `python train.py --dataset inD --dynamic-edges true --seed <s>` ·
evaluate with matching flags · figures via `visualize_comparison.py`, `compare_runs.py`.
Weights and per-run metrics: `saved_models/inD/`, `results/`.

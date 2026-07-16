"""Side-by-side trajectory comparison: baseline vs dynamic-edges MTP-GO.

Renders the same test scenes (as picked in visualize_predictions.py) with the
baseline checkpoint (left) and the --dynamic-edges checkpoint (right).
Per-panel titles carry each model's per-scene most-likely-mode ADE.

Usage:
    python visualize_comparison.py --dataset inD
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.lines import Line2D

from visualize_predictions import (args, build_model, predict_scene, scene_meta,
                                   plot_scene, TrajectoryPredictionDataset,
                                   C_HIST, C_GT, C_PRED, INK, INK_2, SURFACE, SEARCH_PATH)
from datamodule import MetaInfo  # meta .pt files unpickle __main__.MetaInfo
import torch
from lightning.pytorch import seed_everything

# Same scenes as the single-model figure (picked by dynedge per-scene ADE spectrum)
SCENES = [(747, 'easy'), (617, 'easy'), (31, 'median'),
          (807, 'median'), (0, 'hard'), (413, 'hard')]

VARIANTS = [
    ('baseline', dict(dynamic_edges=False, full_edges=False, add_name='')),
    ('dynamic edges', dict(dynamic_edges=True, full_edges=False, add_name='_DYNEDGE')),
]


def main():
    seed_everything(args.seed, workers=True)
    device = 'cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu'

    models = []
    for label, overrides in VARIANTS:
        args.dynamic_edges = overrides['dynamic_edges']
        args.full_edges = overrides['full_edges']
        args.add_name = overrides['add_name']
        model, name = build_model()
        dataset = TrajectoryPredictionDataset('testing', args.dataset,
                                              full_edges=args.full_edges)
        models.append((label, model.to(device), dataset))
        print(f'loaded {label}: {name}')

    backgrounds = {}
    n_rows = len(SCENES) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.6 * n_rows), facecolor=SURFACE)

    for si, (idx, lab) in enumerate(SCENES):
        meta = scene_meta(idx)
        if meta.rec_id not in backgrounds:
            try:
                img = mpimg.imread(f'{SEARCH_PATH}/data/{meta.rec_id}_background.png')
                rm = pd.read_csv(f'{SEARCH_PATH}/data/{meta.rec_id}_recordingMeta.csv')
                backgrounds[meta.rec_id] = (img, float(rm.orthoPxToMeter[0]) * 12)
            except Exception:
                backgrounds[meta.rec_id] = None
        row, col0 = divmod(si, 2)
        for mi, (mlabel, model, dataset) in enumerate(models):
            ax = axes[row][col0 * 2 + mi]
            scene = predict_scene(model, dataset, idx, device)
            plot_scene(ax, scene, meta, backgrounds)
            ax.set_title(f'scene {idx} ({lab}) - {mlabel}: ADE {scene["ade"]:.2f} m',
                         color=INK, fontsize=10, loc='left')

    handles = [Line2D([], [], color=C_HIST, lw=1.6, label='observed history (3 s)'),
               Line2D([], [], color=C_GT, lw=2, label='ground-truth future (5 s)'),
               Line2D([], [], color=C_PRED, lw=2, label='prediction (most-likely mode)'),
               Line2D([], [], color=C_PRED, lw=1, alpha=0.35, label='other modes (alpha = weight)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=11, labelcolor=INK_2)
    fig.suptitle('MTP-GO on inD: baseline vs dynamic edges - same test scenes',
                 color=INK, fontsize=14, fontweight='bold', x=0.05, ha='left')
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = 'trajectories_baseline_vs_dynedge.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

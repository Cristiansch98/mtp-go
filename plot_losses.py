"""Visualize MTP-GO training/validation metrics from Lightning CSVLogger output.

Usage:
    python plot_losses.py                      # auto-discover latest logs/*/version_*/metrics.csv
    python plot_losses.py --csv path/to/metrics.csv --out training_curves.png

Produces a 2x2 panel figure (one y-scale per panel):
    1. train_loss vs. step (raw + EMA smoothing)
    2. val_nll vs. epoch
    3. displacement errors (val_ade, val_fde, val_tv_ade, val_tv_fde) vs. epoch
    4. mode_dist vs. epoch
and prints a summary of best validation metrics.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Validated categorical palette (light mode), fixed slot order
SERIES = ['#2a78d6', '#008300', '#e87ba4', '#eda100']
SURFACE = '#fcfcfb'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'


def find_latest_csv():
    candidates = glob.glob('logs/**/metrics.csv', recursive=True)
    if not candidates:
        raise FileNotFoundError('No metrics.csv found under logs/ - run training with --use-logger true')
    return max(candidates, key=os.path.getmtime)


def ema(series, alpha=0.05):
    return series.ewm(alpha=alpha).mean()


def style_axis(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc='left', fontweight='bold')
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='path to Lightning metrics.csv')
    parser.add_argument('--out', type=str, default='training_curves.png', help='output figure path')
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv()
    print(f'Reading metrics from: {csv_path}')
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    fig.suptitle('MTP-GO training on inD', color=INK, fontsize=13, fontweight='bold', x=0.06, ha='left')

    # Panel 1: training loss vs. epoch (logged per-epoch by base_mdn.py)
    ax = axes[0][0]
    if 'train_loss' in df.columns:
        tr = df[['epoch', 'train_loss']].dropna().groupby('epoch', as_index=False).last()
        ax.plot(tr.epoch, tr.train_loss, color=SERIES[0], linewidth=1, alpha=0.3)
        ax.plot(tr.epoch, ema(tr.train_loss, alpha=0.3), color=SERIES[0], linewidth=2, label='train_loss (EMA)')
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    style_axis(ax, 'Training loss (WNLL, objective-scheduled)', 'epoch', 'loss')

    # Panel 2: validation NLL vs. epoch
    ax = axes[0][1]
    if 'val_nll' in df.columns:
        va = df[['epoch', 'val_nll']].dropna().groupby('epoch', as_index=False).last()
        ax.plot(va.epoch, va.val_nll, color=SERIES[0], linewidth=2)
        best = va.loc[va.val_nll.idxmin()]
        ax.scatter([best.epoch], [best.val_nll], s=36, color=SERIES[0], zorder=3)
        ax.annotate(f'best {best.val_nll:.3f} @ ep {int(best.epoch)}', (best.epoch, best.val_nll),
                    textcoords='offset points', xytext=(-6, 10), fontsize=8, color=INK_2, ha='right')
    style_axis(ax, 'Validation NLL', 'epoch', 'NLL')

    # Panel 3: displacement errors vs. epoch (same unit: meters)
    ax = axes[1][0]
    disp_metrics = [m for m in ('val_ade', 'val_fde', 'val_tv_ade', 'val_tv_fde') if m in df.columns]
    for i, m in enumerate(disp_metrics):
        va = df[['epoch', m]].dropna().groupby('epoch', as_index=False).last()
        ax.plot(va.epoch, va[m], color=SERIES[i], linewidth=2, label=m)
        # direct label at line end (relief for low-contrast slots)
        ax.annotate(m, (va.epoch.iloc[-1], va[m].iloc[-1]), textcoords='offset points',
                    xytext=(5, 4 if i % 2 == 0 else -4), fontsize=8, color=INK_2, va='center')
    if disp_metrics:
        ax.margins(x=0.14)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    style_axis(ax, 'Validation displacement errors', 'epoch', 'error (m)')

    # Panel 4: mode distance vs. epoch
    ax = axes[1][1]
    if 'mode_dist' in df.columns:
        va = df[['epoch', 'mode_dist']].dropna().groupby('epoch', as_index=False).last()
        ax.plot(va.epoch, va.mode_dist, color=SERIES[0], linewidth=2)
    style_axis(ax, 'Mixture mode distance', 'epoch', 'distance (m)')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f'Saved figure to: {args.out}')

    # Summary of best validation metrics
    print('\n--- Best validation metrics ---')
    for m in ('val_nll', 'val_ade', 'val_fde', 'val_tv_ade', 'val_tv_fde', 'mode_dist'):
        if m in df.columns:
            va = df[['epoch', m]].dropna().groupby('epoch', as_index=False).last()
            if len(va):
                best = va.loc[va[m].idxmin()]
                print(f'{m:12s}  best {best[m]:8.4f}  @ epoch {int(best.epoch):4d}  (last {va[m].iloc[-1]:8.4f})')


if __name__ == '__main__':
    main()

"""Compare training/validation metric histories across multiple MTP-GO runs.

Usage:
    python compare_runs.py --runs baseline=results/runs/baseline_metrics.csv \
                                  M4=results/runs/M4_metrics.csv \
                           --out results/comparison.png

Panels (one y-scale each): train_loss, val_nll, val_ade, val_fde.
One colored series per run; prints a best-metrics table to stdout.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Validated categorical palette (light mode), fixed slot order
SERIES = ['#2a78d6', '#008300', '#e87ba4', '#eda100', '#1baf7a', '#eb6834']
SURFACE = '#fcfcfb'
GRID = '#e1e0d9'
BASELINE_C = '#c3c2b7'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'

PANELS = [
    ('train_loss', 'Training loss (WNLL, objective-scheduled)', 'loss'),
    ('val_nll', 'Validation NLL', 'NLL'),
    ('val_ade', 'Validation ADE', 'error (m)'),
    ('val_fde', 'Validation FDE', 'error (m)'),
]


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
        ax.spines[side].set_color(BASELINE_C)


def per_epoch(df, metric):
    if metric not in df.columns:
        return None
    va = df[['epoch', metric]].dropna().groupby('epoch', as_index=False).last()
    return va if len(va) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='NAME=path/to/metrics.csv entries')
    parser.add_argument('--out', type=str, default='comparison.png')
    args = parser.parse_args()

    runs = []
    for entry in args.runs:
        name, _, path = entry.partition('=')
        runs.append((name, pd.read_csv(path)))
    if len(runs) > len(SERIES):
        raise SystemExit(f'At most {len(SERIES)} runs supported (palette slots)')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    fig.suptitle('MTP-GO on inD: run comparison', color=INK, fontsize=13,
                 fontweight='bold', x=0.06, ha='left')

    for ax, (metric, title, ylabel) in zip(axes.flat, PANELS):
        plotted = False
        for i, (name, df) in enumerate(runs):
            va = per_epoch(df, metric)
            if va is None:
                continue
            ax.plot(va.epoch, va[metric], color=SERIES[i], linewidth=2, label=name)
            # direct label at line end (relief for low-contrast slots)
            ax.annotate(name, (va.epoch.iloc[-1], va[metric].iloc[-1]),
                        textcoords='offset points', xytext=(5, 4 if i % 2 == 0 else -4),
                        fontsize=8, color=INK_2, va='center')
            plotted = True
        if plotted:
            ax.margins(x=0.15)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        style_axis(ax, title, 'epoch', ylabel)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f'Saved figure to: {args.out}')

    print('\n--- Best validation metrics per run ---')
    header = f'{"run":12s}' + ''.join(f'{m:>12s}' for m, _, _ in PANELS[1:])
    print(header)
    for name, df in runs:
        row = f'{name:12s}'
        for metric, _, _ in PANELS[1:]:
            va = per_epoch(df, metric)
            row += f'{va[metric].min():12.4f}' if va is not None else f'{"-":>12s}'
        print(row)


if __name__ == '__main__':
    main()

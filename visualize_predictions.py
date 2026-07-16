"""Visualize multi-agent trajectory predictions of a trained MTP-GO model.

Renders test-set scenes over the inD background image: observed history,
ground-truth future, all mixture modes (alpha = mixture weight), the
most-likely mode highlighted, and 1-sigma position-uncertainty ellipses
at 1s / 3s / 5s. Scenes are chosen across the difficulty spectrum
(easy / median / hard by per-scene ADE of the most-likely mode).

Usage (flags must match the checkpoint, like test.py):
    python visualize_predictions.py --dataset inD --dynamic-edges true --add-name _DYNEDGE
"""
import os.path
import warnings
from argument_parser import args
from base_mdn import *
from datamodule import *
from models.gru_gnn import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from torch_geometric.data import Batch
from lightning.pytorch import seed_everything

warnings.filterwarnings('ignore')

# Validated palette / chrome (light mode)
C_PRED = '#2a78d6'   # most-likely mode + ellipses
C_GT = '#008300'     # ground-truth future
C_HIST = '#0b0b0b'   # observed history
INK, INK_2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'

SEARCH_PATH = '../data_sets/inD'
N_EASY, N_MED, N_HARD = 2, 2, 2
ELLIPSE_STEPS = (4, 14, 24)  # 1 s, 3 s, 5 s at 5 Hz


def build_model():
    if args.dataset == 'highD':
        input_len, v_types = 2, 2
    elif args.dataset == 'rounD':
        input_len, v_types = 3, 7
    else:
        input_len, v_types = 3, 4
    n_features = 9
    static_f_dim = v_types * int(args.n_ode_static)
    dt = 2e-1
    max_l = int(input_len * (1 / dt)) + 1

    if args.motion_model == 'neuralode':
        m_model = FirstOrderNeuralODE(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures,
                                      static_f_dim=static_f_dim, n_hidden=args.n_ode_hidden,
                                      n_layers=args.n_ode_layers)
    else:
        m_model = SecondOrderNeuralODE(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures,
                                       static_f_dim=static_f_dim, n_hidden=args.n_ode_hidden,
                                       n_layers=args.n_ode_layers)
    save_name = type(m_model).__name__
    d_str = args.dataset
    de_str = "DE" if args.dynamic_edges else ""
    full_save_name = f"{save_name}{args.hidden_size}G{args.n_gnn_layers}{de_str}{d_str[0].upper() + d_str[1:]}{args.add_name}"

    encoder = GRUGNNEncoder(input_size=n_features, hidden_size=args.hidden_size,
                            n_mixtures=m_model.mixtures, n_layers=args.n_gnn_layers,
                            gnn_layer=args.gnn_layer, n_heads=args.n_heads,
                            static_f_dim=static_f_dim, init_static=args.init_static,
                            use_edge_features=args.use_edge_features)
    decoder = GRUGNNDecoder(m_model, hidden_size=encoder.hidden_size, max_length=max_l,
                            n_layers=args.n_gnn_layers, n_heads=args.n_heads,
                            static_f_dim=static_f_dim, gnn_layer=args.gnn_layer,
                            init_static=args.init_static, dynamic_edges=args.dynamic_edges)

    ckpt = f"saved_models/{args.dataset}/{full_save_name}.ckpt"
    model = LitEncoderDecoder.load_from_checkpoint(ckpt, encoder=encoder, decoder=decoder,
                                                   args=args, weights_only=False)
    model.eval()
    return model, full_save_name


@torch.no_grad()
def predict_scene(model, dataset, idx, device):
    data = Batch.from_data_list([dataset.get(idx)]).to(device)
    all_states, all_Ps, mixture_coeffs, real_mask, target = model.encode_decode(data, 0)
    pi = torch.softmax(mixture_coeffs, dim=-1)          # (N, M)
    ml = torch.argmax(pi, dim=-1)                       # (N,)
    n = all_states.shape[0]
    ml_traj = all_states[torch.arange(n), :, ml, :2]    # (N, T, 2)
    ml_P = all_Ps[torch.arange(n), :, ml, :2, :2]       # (N, T, 2, 2)
    norm = torch.linalg.norm(ml_traj - target[..., :2], dim=-1)  # (N, T)
    mask = real_mask                                    # (N, T)
    ade = float((norm * mask).sum() / mask.sum().clamp(min=1))
    obs = data.x[..., :2].cpu()                         # (N, T_obs, 2)
    gt = target[..., :2].cpu()
    return dict(ade=ade, obs=obs, gt=gt, mask=mask.cpu(),
                states=all_states[..., :2].cpu(), pi=pi.cpu(),
                ml_traj=ml_traj.cpu(), ml_P=ml_P.cpu(), n_agents=n)


def scene_meta(idx):
    meta = torch.load(f'data/{args.dataset}-gnn/testing/meta/dat{idx}.pt', weights_only=False)
    return meta


def draw_ellipse(ax, mean, cov, color):
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp(min=1e-9)
    angle = float(torch.rad2deg(torch.atan2(evecs[1, 1], evecs[0, 1])))
    w, h = 2 * float(evals[1].sqrt()), 2 * float(evals[0].sqrt())  # 1-sigma
    ax.add_patch(Ellipse(mean, w, h, angle=angle, facecolor=color, alpha=0.15,
                         edgecolor=color, linewidth=0.8))


def plot_scene(ax, scene, meta, backgrounds):
    rec, p0 = meta.rec_id, meta.initial_pos
    bg = backgrounds.get(rec)
    if bg is not None:
        img, ortho = bg
        # inD convention: x_px = x_m / ortho, y_px = -y_m / ortho (y is negative)
        extent = [-p0[0], (img.shape[1] * ortho) - p0[0],
                  -(img.shape[0] * ortho) - p0[1], -p0[1]]
        ax.imshow(img, extent=extent, alpha=0.65, zorder=0)

    xs, ys = [], []
    for a in range(scene['n_agents']):
        obs, gt, mask = scene['obs'][a], scene['gt'][a], scene['mask'][a]
        valid_obs = ~(obs == 0).all(dim=-1)
        t_valid = mask.bool()
        if t_valid.sum() < 2:
            continue
        # All mixture modes, alpha proportional to weight
        for m in range(scene['states'].shape[2]):
            w = float(scene['pi'][a, m])
            if w < 0.02:
                continue
            tr = scene['states'][a, :, m]
            ax.plot(tr[t_valid, 0], tr[t_valid, 1], color=C_PRED, linewidth=1,
                    alpha=min(0.85, 0.15 + w), zorder=3)
        ml = scene['ml_traj'][a]
        ax.plot(obs[valid_obs, 0], obs[valid_obs, 1], color=C_HIST, linewidth=1.6,
                zorder=4, solid_capstyle='round')
        ax.plot(gt[t_valid, 0], gt[t_valid, 1], color=C_GT, linewidth=2, zorder=5)
        ax.plot(ml[t_valid, 0], ml[t_valid, 1], color=C_PRED, linewidth=2, zorder=6)
        ax.scatter(obs[valid_obs, 0][-1], obs[valid_obs, 1][-1], s=18, color=C_HIST, zorder=7)
        for st in ELLIPSE_STEPS:
            if st < len(t_valid) and t_valid[st]:
                draw_ellipse(ax, ml[st].tolist(), scene['ml_P'][a, st], C_PRED)
        xs += [float(obs[valid_obs, 0].min()), float(obs[valid_obs, 0].max()),
               float(gt[t_valid, 0].min()), float(gt[t_valid, 0].max())]
        ys += [float(obs[valid_obs, 1].min()), float(obs[valid_obs, 1].max()),
               float(gt[t_valid, 1].min()), float(gt[t_valid, 1].max())]
    if xs:
        m = 8.0
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect('equal')
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#c3c2b7')


def main():
    seed_everything(args.seed, workers=True)
    device = 'cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu'
    model, name = build_model()
    model = model.to(device)
    dataset = TrajectoryPredictionDataset('testing', args.dataset)
    n = dataset.len()
    print(f'Model {name} on {device}; scoring {n} test scenes...')

    scored = []
    for i in range(n):
        try:
            s = predict_scene(model, dataset, i, device)
            if s['n_agents'] >= 3 and float(torch.linalg.norm(s['gt'][0, -1] - s['gt'][0, 0])) > 3.0:
                scored.append((i, s['ade']))
        except Exception:
            continue
    scored.sort(key=lambda t: t[1])
    print(f'{len(scored)} candidate scenes (>=3 agents, moving); '
          f'ADE range {scored[0][1]:.2f}-{scored[-1][1]:.2f} m')

    mid = len(scored) // 2
    picks = scored[:N_EASY] + scored[mid - 1:mid + N_MED - 1] + scored[-N_HARD:]
    labels = ['easy'] * N_EASY + ['median'] * N_MED + ['hard'] * N_HARD

    backgrounds = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=SURFACE)
    for ax, (idx, ade), lab in zip(axes.flat, picks, labels):
        meta = scene_meta(idx)
        if meta.rec_id not in backgrounds:
            try:
                img = mpimg.imread(f'{SEARCH_PATH}/data/{meta.rec_id}_background.png')
                rm = pd.read_csv(f'{SEARCH_PATH}/data/{meta.rec_id}_recordingMeta.csv')
                backgrounds[meta.rec_id] = (img, float(rm.orthoPxToMeter[0]) * 12)
            except Exception as e:
                print(f'no background for rec {meta.rec_id}: {e}')
                backgrounds[meta.rec_id] = None
        scene = predict_scene(model, dataset, idx, device)
        plot_scene(ax, scene, meta, backgrounds)
        ax.set_title(f'scene {idx} ({lab}) - {scene["n_agents"]} agents, ADE {ade:.2f} m',
                     color=INK, fontsize=10, loc='left')

    handles = [Line2D([], [], color=C_HIST, lw=1.6, label='observed history (3 s)'),
               Line2D([], [], color=C_GT, lw=2, label='ground-truth future (5 s)'),
               Line2D([], [], color=C_PRED, lw=2, label='prediction (most-likely mode)'),
               Line2D([], [], color=C_PRED, lw=1, alpha=0.35, label='other modes (alpha = weight)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=10, labelcolor=INK_2)
    fig.suptitle(f'MTP-GO predictions on inD test scenes - {name}',
                 color=INK, fontsize=13, fontweight='bold', x=0.06, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = f'trajectories_{name}.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

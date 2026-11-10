# 電線をまたぐ組の経路を横から見た図にする（研究メモ §9bb）。横軸＝経路に沿った水平距離、縦軸＝地面からの高さ。
# 経路の真上・真下（水平距離 0.3m 以内）にある電線の点（wire3d.py の復元）を重ねる。始点・終点（指定した位置）とまっすぐ結んだ線も描く。
# 使い方: python3 real/plot_cross_paths.py <exp> <model>:<pair>[:<add_wires 0|1>] [...]  → research/wirebench/figs/cross_paths_<exp>.png
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402
import numpy as np                                    # noqa: E402
import pycolmap                                       # noqa: E402
from scipy.spatial import cKDTree                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_splatnav import camera_frame               # noqa: E402

FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=FONT).get_name()]


def densify(T, step=0.02):
    seg = np.linalg.norm(np.diff(T, axis=0), axis=1)
    return np.concatenate([np.linspace(a, b, max(2, int(np.ceil(d / step)) + 1)) for a, b, d in zip(T[:-1], T[1:], seg)])


exp = Path(sys.argv[1]).resolve()
specs = sys.argv[2:]
_, C, _ = camera_frame(pycolmap.Reconstruction(str(exp / "sparse" / "0")), 1.4)
ground = C[:, 2].mean() - 1.4
W = np.load(exp / "wire3d_bright.npz")["points"]
wt2 = cKDTree(W[:, :2])
pairs = np.load(exp / "splatnav_cross_pairs.npz")["pairs"]
rows = {(r["model"], r["pair"], r["add_wires"]): r for r in map(json.loads, open(exp / "splatnav_cross_traj.jsonl")) if r.get("status") == "ok"}
an = {}
import pandas as pd                                    # noqa: E402
for _, r in pd.read_csv(exp / "splatnav_cross_traj_analysis.csv").iterrows():
    an[(r.model, int(r.pair), bool(r.add_wires))] = r
fig, axs = plt.subplots(len(specs), 1, figsize=(12, 3.6 * len(specs)))
axs = np.atleast_1d(axs)
for ax, sp in zip(axs, specs):
    m, p, *w = sp.split(":")
    p = int(p)
    aw = bool(int(w[0])) if w else False
    r = rows[(m, p, aw)]
    T = densify(np.load(exp / r["traj_file"])[:, :3])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T[:, :2], axis=0), axis=1))]
    nb = wt2.query_ball_point(T[:, :2], 0.3)
    ws, wz = [], []
    for i, idx in enumerate(nb):
        for j in idx:
            ws.append(s[i])
            wz.append(W[j, 2] - ground)
    ax.scatter(ws, wz, s=4, c="black", label="電線の点（経路の真上・真下 0.3m 以内）")
    ax.plot(s, T[:, 2] - ground, color="tab:blue", lw=2, label="Splat-Nav の経路")
    a0, g0 = pairs[p]
    ax.plot([0, s[-1]], [a0[2] - ground, g0[2] - ground], "--", color="gray", label="始点と終点をまっすぐ結んだ線（参考）")
    k = an[(m, p, aw)]
    ax.set_title(f"{m}・組{p}{'（電線を地図に足した）' if aw else ''}：{k.kind}、電線との最小距離 {k.min_dist_wire_m:.2f}m、経路長は直線の {k.len_ratio:.2f} 倍", fontsize=12)
    ax.set_xlabel("経路に沿った水平距離 [m]")
    ax.set_ylabel("地面からの高さ [m]")
    ax.legend(fontsize=9, loc="upper right")
out = HERE.parent / "figs" / f"cross_paths_{exp.name}.png"
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out)

# 電線スパン3cm以内のガウシアン不透明度分布の図を生成する（残差解剖パートB）
# analyze_gaussians.py が保存する gaussian_analysis_<model>_<it>.npz を読む。
# 2026-09-24（監査R-06）：左＝ply の生の不透明度、右＝Mip の 3Dフィルタ込み実効不透明度（レンダラが使う値）。
# 旧図（生の値のみ）の「Mipで不透明側へ大移動」は生の値の見かけで、実効値では Mip の方が透明寄り。
# 使い方: python3 plot_opacity_hist.py   → figs/wire_opacity_hist.png

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = Path("/workspace/experiments")
HERE = Path(__file__).parent
RUNS = [("3DGS", "20260704_wb_r15mm_f80_open", "output", "#d95f02", "--"),
        ("3DGS+inject", "20260710_wb_r15mm_f80_open_inject", "output", "#7570b3", "--"),
        ("Mip", "20260704_wb_r15mm_f80_open", "output_mip", "#d95f02", "-"),
        ("Mip+inject", "20260710_wb_r15mm_f80_open_inject", "output_mip", "#7570b3", "-")]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
bins = np.linspace(0, 1, 41)
for ax, key, title in ((axes[0], "opacity_raw_3cm", "raw opacity in ply (sigmoid)"),
                       (axes[1], "opacity_3cm", "effective opacity used by renderer (Mip: x 3D-filter coef)")):
    for label, d, m, color, ls in RUNS:
        z = np.load(E / d / "gt" / f"gaussian_analysis_{m}_7000.npz")
        op = z[key]
        ax.hist(op, bins=bins, histtype="step", density=True, lw=2, color=color, ls=ls,
                label=f"{label} (n={len(op)}, q50={np.median(op):.3f}, >=0.5: {(op >= 0.5).sum()})")
    ax.set_xlabel(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, ls=":")
axes[0].set_ylabel("density")
fig.suptitle("gaussians within 3cm of wire spans (polyline distance; r15mm, open, 80 views, 7k)")
fig.tight_layout()
out = HERE / "figs" / "wire_opacity_hist.png"
fig.savefig(out, dpi=110)
print(f"保存: {out}")

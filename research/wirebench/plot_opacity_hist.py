# 電線スパン3cm以内のガウシアン不透明度分布の図を生成する（残差解剖パートB）
# analyze_gaussians.py が保存する gaussian_analysis_<model>_<it>.npz を読む。
# 「素の3DGSでは透明のまま／Mipで不透明側へ大移動」を1枚で示すのが目的。
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

fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(0, 1, 41)
for label, d, m, color, ls in RUNS:
    z = np.load(E / d / "gt" / f"gaussian_analysis_{m}_7000.npz")
    op = z["opacity_3cm"]
    ax.hist(op, bins=bins, histtype="step", density=True, lw=2,
            color=color, ls=ls, label=f"{label} (n={len(op)}, q50={np.median(op):.2f})")
ax.set_xlabel("opacity (sigmoid)")
ax.set_ylabel("density")
ax.set_yscale("log")
ax.set_title("opacity of gaussians within 3cm of wire spans (r15mm, open, 80 views)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, ls=":")
fig.tight_layout()
out = HERE / "figs" / "wire_opacity_hist.png"
fig.savefig(out, dpi=110)
print(f"保存: {out}")

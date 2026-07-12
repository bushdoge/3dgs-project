# Mipスイープ（Phase 1-7）の図を mip_sweep_results.csv から生成する。
# 第1波（figs/sweep_wave1_radius.png）と同じ配色（open=橙 / enclosed=緑）で、
# 手法を線種（3DGS=破線 / Mip=実線）で重ね描きした3面図。
# アンカー条件（r15mm open）の run2 は同色の×印で重ねて分散の目安を示す。
# 使い方: python3 plot_mip_sweep.py   → figs/mip_sweep_radius.png

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "mip_sweep_results.csv")))
for r in rows:
    for k in ["wire_radius_m", "psnr_wire", "gap_db", "disappearance_rate", "run"]:
        r[k] = float(r[k])

COL = {"open": "#d95f02", "enclosed": "#1b9e77"}
LS = {"3dgs": "--", "mip": "-"}
PANELS = [("psnr_wire", "wire PSNR [dB]", "wire PSNR vs radius"),
          ("gap_db", "gap [dB]", "wire-vs-bg gap"),
          ("disappearance_rate", "disappearance rate", "disappearance (1 - norm. recall)")]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (key, ylab, title) in zip(axes, PANELS):
    for variant in ["open", "enclosed"]:
        for method in ["3dgs", "mip"]:
            pts = sorted([(r["wire_radius_m"] * 1000, r[key]) for r in rows
                          if r["variant"] == variant and r["method"] == method
                          and r["run"] == 1])
            ax.plot(*zip(*pts), LS[method], marker="o", color=COL[variant],
                    alpha=1.0 if method == "mip" else 0.55,
                    label=f"{variant} {method.upper() if method == '3dgs' else 'Mip'}")
    for r in rows:  # アンカー条件のrun2（分散の目安）
        if r["run"] == 2:
            ax.plot(r["wire_radius_m"] * 1000, r[key], "x", ms=10, mew=2.5,
                    color=COL[r["variant"]],
                    alpha=1.0 if r["method"] == "mip" else 0.55)
    ax.set_xlabel("wire radius [mm]")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3, ls=":")
axes[0].legend(fontsize=8)
fig.suptitle("wirebench Mip sweep: radius x background (80 views, 7k iters; x = rerun of anchor condition)")
fig.tight_layout()
out = HERE / "figs" / "mip_sweep_radius.png"
fig.savefig(out, dpi=110)
print(f"保存: {out}")

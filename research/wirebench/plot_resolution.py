# 解像度統制（§9aj）の図を resolution_results.csv から生成する。
# 旧主張（2026-09-07）：見かけ画素幅が3px→1.5pxになると学習延長が無効になり（−0.10dB）、AAの寄与が上がる。
# 2026-09-24 監査R-14：これは PSNR マスクの2px膨張（md2）でだけ出る。膨張なし（md0）では1.5pxでも学習延長は
#   効き、AAの寄与は上がらない。本図は md2 と md0 を並べ、各寄与に95%区間（run間分散から）を付ける。
#   データは collect_resolution.py が作る resolution_results.csv。
# 幾何・シーン内容・マスク面積を完全固定したまま見かけ幅だけを半分にした統制なので、
# 太さスイープ（フェンス棒混合の交絡あり・§9ab）より綺麗な軸。
# 使い方: python3 plot_resolution.py   → figs/resolution_ladder.png
#
# 注意: 解像度が違うと絶対PSNRは比較できない（低解像度は指標を水増しする。§9ag）。
#       そのため左パネルは各解像度**内**の差分（寄与）を描き、絶対値は右パネルに参考として置く。

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "resolution_results.csv")))

def collect(col):
    v = defaultdict(list)          # (px, model, iters) -> [PSNR]
    for r in rows:
        if r.get(col):
            v[(float(r["wire_px_p50"]), r["model"], int(r["iters"]))].append(float(r[col]))
    return v


def diff(v, a, b):
    """平均の差 b−a と 95% 区間の半幅（Welch 近似。n=1 の群は他群の分散を借りる）"""
    xa, xb = np.array(v[a]), np.array(v[b])
    pooled = np.concatenate([x - x.mean() for x in (xa, xb) if len(x) > 1])
    sd = pooled.std(ddof=1) if len(pooled) > 1 else np.nan
    va = xa.var(ddof=1) if len(xa) > 1 else sd ** 2
    vb = xb.var(ddof=1) if len(xb) > 1 else sd ** 2
    return xb.mean() - xa.mean(), 1.96 * np.sqrt(va / len(xa) + vb / len(xb)), len(xa), len(xb)


WIDTHS = [3.0, 1.51]
LABEL = {3.0: "3.0 px\n(1024x768)", 1.51: "1.5 px\n(512x384)"}
COL = {"train": "#7570b3", "aa": "#d95f02"}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
summary = {}
for ax, col, title in ((axes[0], "psnr_wireonly", "mask dilated 2px (md2, old metric)"),
                       (axes[1], "psnr_wireonly_md0", "mask NOT dilated (md0)")):
    v = collect(col)
    x = np.arange(len(WIDTHS))
    tr = [diff(v, (p, "3dgs", 7000), (p, "3dgs", 30000)) for p in WIDTHS]
    aa = [diff(v, (p, "3dgs", 30000), (p, "mip_ks0.1", 30000)) for p in WIDTHS]
    ax.bar(x - 0.18, [t[0] for t in tr], 0.34, yerr=[t[1] for t in tr], capsize=4,
           label="training 7k->30k", color=COL["train"])
    ax.bar(x + 0.18, [a[0] for a in aa], 0.34, yerr=[a[1] for a in aa], capsize=4,
           label="anti-aliasing (Mip)", color=COL["aa"])
    for xi, (t, a) in enumerate(zip(tr, aa)):
        ax.text(xi - 0.18, max(t[0] + t[1], 0) + 0.12, f"{t[0]:+.2f}\n(n={t[2]}/{t[3]})", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.18, max(a[0] + a[1], 0) + 0.12, f"{a[0]:+.2f}\n(n={a[2]}/{a[3]})", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[p] for p in WIDTHS])
    ax.set_ylabel("wire-only PSNR gain [dB] (95% CI)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(-1.5, ax.get_ylim()[0]), ax.get_ylim()[1] + 1.0)
    summary[col] = (tr, aa)

# ── 右：絶対値の梯子（md0。参考。解像度をまたいだ比較は不可） ─────────────────
ax = axes[2]
v = collect("psnr_wireonly_md0")
STEPS = [("3dgs", 7000, "3DGS 7k"), ("3dgs", 30000, "3DGS 30k"),
         ("mip_ks0.1", 30000, "Mip 30k"), ("mip_ks0.05", 30000, "Mip 30k\nks0.05")]
for p, mk in zip(WIDTHS, ["o-", "s--"]):
    y = [np.mean(v[(p, m, it)]) if v.get((p, m, it)) else np.nan for m, it, _ in STEPS]
    ax.plot(range(len(STEPS)), y, mk, label=LABEL[p].replace("\n", " "))
ax.set_xticks(range(len(STEPS))); ax.set_xticklabels([s_[2] for s_ in STEPS], fontsize=9)
ax.set_ylabel("wire-only PSNR, md0 [dB]")
ax.set_title("Absolute ladder, md0 (do NOT compare across resolutions)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.tight_layout()
out = HERE / "figs" / "resolution_ladder.png"
fig.savefig(out, dpi=150)
print(f"saved: {out}")
for col, (tr, aa) in summary.items():
    for p, t, a in zip(WIDTHS, tr, aa):
        print(f"{col:18s} {p}px: train {t[0]:+.2f}±{t[1]:.2f} dB (n={t[2]}/{t[3]}), AA {a[0]:+.2f}±{a[1]:.2f} dB (n={a[2]}/{a[3]})")

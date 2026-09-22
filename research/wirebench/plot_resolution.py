# 解像度統制（§9aj）の図を resolution_results.csv から生成する。
# 主張：見かけ画素幅が3px→1.5pxになると「学習量を増やす」処方は無効になり（+2.83→−0.10dB）、
#       代わりにAA（Mip）の寄与が上がる（+3.01→+4.03dB）＝主犯はサンプリング。
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

vals = defaultdict(list)          # (px, model, iters) -> [wire-only PSNR]
for r in rows:
    if not r["psnr_wireonly"]:
        continue
    vals[(float(r["wire_px_p50"]), r["model"], int(r["iters"]))].append(float(r["psnr_wireonly"]))
mean = {k: np.mean(v) for k, v in vals.items()}

WIDTHS = [3.0, 1.51]
LABEL = {3.0: "3.0 px\n(1024x768)", 1.51: "1.5 px\n(512x384)"}
COL = {"train": "#7570b3", "aa": "#d95f02"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# ── 左：各解像度の中での寄与（これが主張） ──────────────────────────────────
ax = axes[0]
x = np.arange(len(WIDTHS))
train = [mean[(p, "3dgs", 30000)] - mean[(p, "3dgs", 7000)] for p in WIDTHS]
aa = [mean[(p, "mip_ks0.1", 30000)] - mean[(p, "3dgs", 30000)] for p in WIDTHS]
ax.bar(x - 0.18, train, 0.34, label="training 7k->30k", color=COL["train"])
ax.bar(x + 0.18, aa, 0.34, label="anti-aliasing (Mip)", color=COL["aa"])
for xi, (t, a) in enumerate(zip(train, aa)):
    # 負の棒は棒の下でなく0線のすぐ上に置く（x軸ラベルと重なるため）
    ax.text(xi - 0.18, t + 0.12 if t >= 0 else 0.12, f"{t:+.2f}", ha="center",
            va="bottom", fontsize=10)
    ax.text(xi + 0.18, a + 0.12, f"{a:+.2f}", ha="center", fontsize=10)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([LABEL[p] for p in WIDTHS])
ax.set_ylabel("wire-only PSNR gain [dB]")
ax.set_title("Which fix works, by apparent wire width")
ax.set_xlabel("apparent wire width (same scene geometry)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ── 右：絶対値の梯子（参考。解像度をまたいだ比較は不可） ───────────────────
ax = axes[1]
STEPS = [("3dgs", 7000, "3DGS 7k"), ("3dgs", 30000, "3DGS 30k"),
         ("mip_ks0.1", 30000, "Mip 30k"), ("mip_ks0.05", 30000, "Mip 30k\nks0.05")]
for p, mk in zip(WIDTHS, ["o-", "s--"]):
    y = [mean.get((p, m, it), np.nan) for m, it, _ in STEPS]
    ax.plot(range(len(STEPS)), y, mk, label=LABEL[p].replace("\n", " "))
ax.set_xticks(range(len(STEPS))); ax.set_xticklabels([s[2] for s in STEPS], fontsize=9)
ax.set_ylabel("wire-only PSNR [dB]")
ax.set_title("Absolute ladder (do NOT compare across resolutions)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.tight_layout()
out = HERE / "figs" / "resolution_ladder.png"
fig.savefig(out, dpi=150)
print(f"saved: {out}")
for p in WIDTHS:
    print(f"{p}px: train {mean[(p,'3dgs',30000)]-mean[(p,'3dgs',7000)]:+.2f} dB, "
          f"AA {mean[(p,'mip_ks0.1',30000)]-mean[(p,'3dgs',30000)]:+.2f} dB")

# 3DGSレンダリング結果の細線領域を評価するスクリプト
# テスト視点ごとに 全体PSNR / 細線領域PSNR / 背景PSNR を計り、
# 細線が消えている様子の比較図（GT | render | 差分）を書き出す。
# 使い方: python3 eval_wires.py <実験dir> <iteration> [モデル出力dir名]
#   例: python3 eval_wires.py . 7000            → <実験dir>/output/test/ours_7000 を評価
#       python3 eval_wires.py . 30000 output_30k → <実験dir>/output_30k/test/ours_30000 を評価

import sys
from pathlib import Path

import cv2
import numpy as np

exp = Path(sys.argv[1])
it = sys.argv[2] if len(sys.argv) > 2 else "7000"
model_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
rdir = exp / model_dir / "test" / f"ours_{it}"
out_dir = exp / "gt" / f"wire_eval_{model_dir}_{it}" if model_dir != "output" else exp / "gt" / f"wire_eval_{it}"
out_dir.mkdir(parents=True, exist_ok=True)

# llffhold=8: テスト視点は名前順の 0,8,16,... 番目
names = sorted(p.stem for p in (exp / "input").glob("*.png"))
test_names = names[::8]

def psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return np.nan
        mse = d[mask].mean()
    else:
        mse = d.mean()
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf

rows = []
kernel = np.ones((5, 5), np.uint8)
for i, name in enumerate(test_names):
    ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
    gt = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
    frame_no = name.split("_")[1]                       # frame_0001 → 0001
    mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{frame_no}.png"), 0)
    if ren is None or gt is None or mask is None:
        continue
    if mask.shape != gt.shape[:2]:
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate((mask > 127).astype(np.uint8), kernel).astype(bool)
    rows.append((name, psnr(ren, gt), psnr(ren, gt, np.repeat(m[..., None], 3, 2)),
                 psnr(ren, gt, np.repeat(~m[..., None], 3, 2)), m.mean() * 100))

print(f"{'view':12s} {'全体PSNR':>9s} {'細線PSNR':>9s} {'背景PSNR':>9s} {'細線px%':>8s}")
for n, p_all, p_wire, p_bg, frac in rows:
    print(f"{n:12s} {p_all:9.2f} {p_wire:9.2f} {p_bg:9.2f} {frac:8.2f}")
arr = np.array([[r[1], r[2], r[3]] for r in rows])
print(f"{'平均':12s} {arr[:,0].mean():9.2f} {arr[:,1].mean():9.2f} {arr[:,2].mean():9.2f}")
print(f"\n→ 背景PSNRと細線PSNRの差: {arr[:,2].mean()-arr[:,1].mean():.2f} dB（大きいほど細線だけ壊れている）")

# 集計用JSON（run_sweep.pyが読む）
import json
json.dump({"iteration": int(it), "model_dir": model_dir,
           "psnr_all": float(arr[:, 0].mean()), "psnr_wire": float(arr[:, 1].mean()),
           "psnr_bg": float(arr[:, 2].mean()),
           "gap_db": float(arr[:, 2].mean() - arr[:, 1].mean()),
           "psnr_wire_min": float(arr[:, 1].min()), "n_test_views": len(rows)},
          open(out_dir / "summary.json", "w"), indent=1)

# ── 比較図: 細線が最も壊れている視点で GT|render|差分 と拡大クロップ ──────────────
worst = int(np.argmin([r[2] for r in rows]))
name = rows[worst][0]
i = worst
ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
gt = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
frame_no = name.split("_")[1]
mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{frame_no}.png"), 0)

full = np.hstack([gt, ren])
cv2.putText(full, "GT", (10, 40), 0, 1.2, (0, 0, 255), 3)
cv2.putText(full, "3DGS render", (gt.shape[1] + 10, 40), 0, 1.2, (0, 0, 255), 3)
cv2.imwrite(str(out_dir / f"compare_{name}.png"), full)

# 細線マスクの重心まわりを2倍拡大でクロップ（上位2クラスタ）
ys, xs = np.where(mask > 127)
if len(ys) > 0:
    for ci, q in enumerate([0.3, 0.7]):
        cy, cx = int(np.quantile(ys, q)), int(np.quantile(xs, q))
        h, w = 140, 240
        y0, x0 = max(cy - h // 2, 0), max(cx - w // 2, 0)
        gtc = gt[y0:y0 + h, x0:x0 + w]
        rnc = ren[y0:y0 + h, x0:x0 + w]
        z = 3
        crop = np.hstack([cv2.resize(gtc, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST),
                          np.full((h * z, 8, 3), 255, np.uint8),
                          cv2.resize(rnc, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)])
        cv2.imwrite(str(out_dir / f"crop{ci}_{name}.png"), crop)
print(f"\n比較図を保存: {out_dir}/（最悪視点 {name}）")

# オラクル点注入（キラー実験用）: GTの電線中心線上の点をCOLMAP点群に人工的に追加する
# 「SfM初期化に電線の種があれば3DGSは細線を再構成できるのか」を直接検証するための前処理。
#   1. COLMAPカメラ中心とGTカメラ中心のUmeyama相似変換を推定（analyze_sfm.pyと同じ手法）
#   2. gt/wire_points.npz の電線中心線サンプルを指定間隔で線形補間して増やす
#   3. GT座標→COLMAP座標へ逆変換し、points3D に追加して <dst>/sparse/0 に書き出す
# 使い方: python3 inject_points.py <src実験dir> <dst実験dir> [--spacing 0.02] [--n-per-wire 60]
#   dst には input/ gt/ が用意済みであること（sparse/0 はこのスクリプトが作る）

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

parser = argparse.ArgumentParser()
parser.add_argument("src", help="元実験dir（sparse/0 と gt/ を読む）")
parser.add_argument("dst", help="注入先実験dir（sparse/0 を書き出す）")
parser.add_argument("--spacing", type=float, default=0.02,
                    help="注入点の間隔[m]（既定0.02=2cm）")
parser.add_argument("--color", type=int, nargs=3, default=[60, 55, 50],
                    help="注入点のRGB（電線の見た目に近い暗灰色）")
args = parser.parse_args()

src, dst = Path(args.src), Path(args.dst)
rec = pycolmap.Reconstruction(str(src / "sparse" / "0"))
gt = json.load(open(src / "gt" / "poses.json"))
wp = np.load(src / "gt" / "wire_points.npz")
wire = wp["points"][wp["labels"] == 1]          # 電線のみ（フェンス棒は注入しない）
n_wires = 3
per = len(wire) // n_wires

# ── Umeyama相似変換（COLMAP→GT）を推定し、その逆でGT点をCOLMAP座標へ ─────────
def umeyama(src_pts, dst_pts):
    mu_s, mu_d = src_pts.mean(0), dst_pts.mean(0)
    sc, dc = src_pts - mu_s, dst_pts - mu_d
    cov = dc.T @ sc / len(src_pts)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (sc ** 2).sum() * len(src_pts)
    t = mu_d - s * R @ mu_s
    return s, R, t

colmap_c, gt_c = [], []
for img in rec.images.values():
    stem = Path(img.name).stem
    if stem in gt["frames"]:
        colmap_c.append(img.projection_center())
        gt_c.append(np.array(gt["frames"][stem])[:3, 3])
s, R, t = umeyama(np.array(colmap_c), np.array(gt_c))
rmse = np.sqrt((((s * (R @ np.array(colmap_c).T)).T + t - np.array(gt_c)) ** 2).sum(1).mean())
print(f"位置合わせ RMSE: {rmse*1000:.1f}mm（スケール {s:.4f}）")
assert rmse < 0.02, "位置合わせが粗すぎる（>2cm）。注入を中止"

# ── 電線中心線を指定間隔で補間 ───────────────────────────────────────────────
inject_gt = []
for k in range(n_wires):
    seg = wire[k * per:(k + 1) * per]
    for a, b in zip(seg[:-1], seg[1:]):
        L = np.linalg.norm(b - a)
        n = max(int(np.ceil(L / args.spacing)), 1)
        for i in range(n):
            inject_gt.append(a + (b - a) * i / n)
    inject_gt.append(seg[-1])
inject_gt = np.array(inject_gt)
# GT→COLMAP 逆変換: x_colmap = Rᵀ (x_gt − t) / s
inject_cm = (R.T @ (inject_gt - t).T).T / s

# ── points3D へ追加して書き出し ─────────────────────────────────────────────
n_before = len(rec.points3D)
color = np.array(args.color, dtype=np.uint8)
for xyz in inject_cm:
    rec.add_point3D(xyz.astype(np.float64), pycolmap.Track(), color)
out = dst / "sparse" / "0"
out.mkdir(parents=True, exist_ok=True)
rec.write(str(out))
print(f"注入完了: 電線中心線 {len(inject_cm)}点（間隔{args.spacing*100:.0f}cm）を追加")
print(f"points3D: {n_before} → {len(rec.points3D)}  出力: {out}")

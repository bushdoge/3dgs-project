# SfM点群における「細線上の点の欠乏」を定量するスクリプト
# COLMAPモデルをGT座標系にUmeyama相似変換で位置合わせし、
# 細線（電線・フェンス棒）近傍のSfM点数を数えて、画素占有率と比較する。
# 使い方: python3 analyze_sfm.py <実験dir>   （<dir>/sparse/0 と <dir>/gt/ を読む）

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

exp = Path(sys.argv[1])
rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
gt = json.load(open(exp / "gt" / "poses.json"))
wp = np.load(exp / "gt" / "wire_points.npz")
wire_pts, wire_labels = wp["points"], wp["labels"]

# ── COLMAPカメラ中心 と GTカメラ中心を画像名で対応付け ─────────────────────────
colmap_centers, gt_centers = [], []
for img in rec.images.values():
    stem = Path(img.name).stem            # frame_0001
    if stem not in gt["frames"]:
        continue
    colmap_centers.append(img.projection_center())
    gt_centers.append(np.array(gt["frames"][stem])[:3, 3])
colmap_centers = np.array(colmap_centers)
gt_centers = np.array(gt_centers)
print(f"登録画像: {len(colmap_centers)} / {len(gt['frames'])}")

# ── Umeyama相似変換（COLMAP座標 → GT/Blender座標）────────────────────────────
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (sc ** 2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return s, R, t

s, R, t = umeyama(colmap_centers, gt_centers)
aligned = (s * (R @ colmap_centers.T)).T + t
rmse = np.sqrt(((aligned - gt_centers) ** 2).sum(1).mean())
print(f"位置合わせ RMSE: {rmse:.4f} m（スケール {s:.4f}）")

# ── SfM点群を変換して細線からの距離を測る ──────────────────────────────────────
pts3d = np.array([p.xyz for p in rec.points3D.values()])
pts3d_w = (s * (R @ pts3d.T)).T + t
tree = cKDTree(wire_pts)
dist, idx = tree.query(pts3d_w)

WIRE_R = 0.015
for th_name, th in [("5cm", 0.05), ("10cm", 0.10), ("20cm", 0.20)]:
    near = dist < th
    n_wire = (wire_labels[idx[near]] == 1).sum()
    n_bar = (wire_labels[idx[near]] == 2).sum()
    print(f"細線から{th_name}以内のSfM点: {near.sum():5d} / {len(pts3d)} "
          f"({near.sum()/len(pts3d)*100:.3f}%)  [電線 {n_wire} / フェンス棒 {n_bar}]")

# ── 厳密版：電線スパン部（電柱から0.3m以上離れた区間）だけで数える ────────────────
# 緩い閾値だと電柱・地面・レール由来の近傍点を拾ってしまうため、
# 「電柱の影響を受けない純粋な電線区間 × 線半径の2倍(3cm)」で測るのが本命の指標。
POLE_XY = np.array([[-4.0, -1.8], [4.0, -1.2]])   # make_scene.py の pole_positions と一致させること
wire_only = wire_pts[wire_labels == 1]
far_from_pole = np.min(np.linalg.norm(wire_only[:, None, :2] - POLE_XY[None], axis=2), axis=1) > 0.3
span = wire_only[far_from_pole]
sdist, _ = cKDTree(span).query(pts3d_w)
print(f"\n[厳密] 電線スパン部GTサンプル: {len(span)}点")
for th in (0.03, 0.05):
    n = (sdist < th).sum()
    print(f"[厳密] 電線スパンから{th*100:.0f}cm以内のSfM点: {n} / {len(pts3d)} ({n/len(pts3d)*100:.4f}%)")

# ── 比較用：細線の画素占有率（GTマスクの平均白率）──────────────────────────────
mask_dir = exp / "gt" / "mask_thin"
fracs = []
for f in sorted(mask_dir.glob("*.png"))[::8]:
    m = cv2.imread(str(f), 0)
    fracs.append((m > 127).mean())
pix_share = np.mean(fracs) * 100
print(f"\n細線の平均画素占有率: {pix_share:.3f}%")
print(f"→ 画素の{pix_share:.2f}%を占める細線に、SfM点の"
      f"{(dist < 0.10).sum()/len(pts3d)*100:.3f}%しか点が無い"
      f"（{pix_share / max((dist<0.10).sum()/len(pts3d)*100, 1e-9):.0f}倍の欠乏）")

# ── 図: SfM点群を1視点に投影して細線の空白を可視化 ─────────────────────────────
name0 = sorted(gt["frames"].keys())[0]
M = np.array(gt["frames"][name0])          # cam2world (Blender/OpenGL)
Rwc, C = M[:3, :3], M[:3, 3]
X_cam = (pts3d_w - C) @ Rwc                # world→cam
# OpenGL(-Z前方) → 画像座標
z = -X_cam[:, 2]
valid = z > 0.1
u = gt["fx"] * X_cam[valid, 0] / z[valid] + gt["cx"]
v = -gt["fy"] * X_cam[valid, 1] / z[valid] + gt["cy"]
img = cv2.imread(str(exp / "input" / f"{name0}.png"))
for x, y in zip(u, v):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
out = exp / "gt" / "sfm_points_overlay.png"
cv2.imwrite(str(out), img)
print(f"\n可視化を保存: {out}（赤=SfM点。細線上に点が無いことを確認）")

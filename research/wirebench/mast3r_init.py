# 検出器を使わない密な再構成（MASt3R）で「SfM は電線上に点を作らない」が検出器に依るかを検定し、
# その点群を初期値にした 3DGS/Mip 学習用の実験dirを作る（監査 memo/review_20260924.md R-18・修正計画C9）。
#   1. 登録済み画像（sparse/0 の images.bin の並び）に MASt3R の sparse global alignment をかける（swin 窓）
#   2. MASt3R のカメラ中心を COLMAP のカメラ中心に Umeyama で合わせ、密な点群を COLMAP 座標へ
#   3. 細線の定量（GT 座標で。analyze_sfm.py と同じ「電線スパン表面 3cm」＋線分距離）：
#      (a) 全密点のうち電線スパン近傍の割合 ÷ 電線スパンの画素占有率 ＝ 欠乏倍率（SfM と同じ定義）
#      (b) GT の電線マスク画素から出た密点のうち、実際に電線表面近くに落ちた割合（深度が電線に乗っているか）
#   4. <dst>/sparse/0 に COLMAP の cameras/images と MASt3R 点群（points3D.ply）を置く
#      → 姿勢・テスト分割・画像は元実験と完全に同一で、初期点群だけが違う実験になる
# 実行（MASt3R の venv で）:
#   /workspace/tools/mast3r/venv/bin/python research/wirebench/mast3r_init.py <src実験dir> <dst実験dir>
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
M3 = Path("/workspace/tools/mast3r")
sys.path.insert(0, str(M3))
sys.path.insert(0, "/opt/gaussian-splatting")

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--winsize", type=int, default=10, help="swin グラフの窓（周回軌道なので cyclic）")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--min-conf", type=float, default=1.5, help="密点の信頼度しきい値（demo 既定）")
ap.add_argument("--n-init", type=int, default=300000, help="初期点群に使う点数（ランダム間引き）")
args = ap.parse_args()
src, dst = Path(args.src).resolve(), Path(args.dst).resolve()

import torch  # noqa: E402
import pycolmap  # noqa: E402
import mast3r.utils.path_to_dust3r  # noqa: F401,E402
from mast3r.model import AsymmetricMASt3R  # noqa: E402
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: E402
from mast3r.image_pairs import make_pairs  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402
from dust3r.utils.device import to_numpy  # noqa: E402
from wbgeom import umeyama, align_colmap_to_gt, apply, wire_span, polyline_distance  # noqa: E402
from eval_wires import load_wire_bar_splitter  # noqa: E402

rec = pycolmap.Reconstruction(str(src / "sparse" / "0"))
ims = sorted(rec.images.values(), key=lambda im: im.name)
files = [str(src / "images" / im.name) for im in ims]
cc = np.array([im.projection_center() for im in ims])

# ── 1. MASt3R ────────────────────────────────────────────────────────────────
out = dst / "mast3r"; out.mkdir(parents=True, exist_ok=True)
model = AsymmetricMASt3R.from_pretrained(
    str(M3 / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")).to("cuda")
imgs = load_images(files, size=args.size, verbose=False)
pairs = make_pairs(imgs, scene_graph=f"swin-{args.winsize}", prefilter=None, symmetrize=True)
print(f"画像 {len(files)} 枚・ペア {len(pairs)}")
scene = sparse_global_alignment(files, pairs, str(out / "cache"), model, lr1=0.07, niter1=300, lr2=0.014, niter2=300,
                                device="cuda", opt_depth=True, shared_intrinsics=True, matching_conf_thr=5.0)
c2w = to_numpy(scene.get_im_poses())                       # (N,4,4)
pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
H, W = imgs[0]["true_shape"][0]
rgb = [((np.asarray(im["img"][0]).transpose(1, 2, 0) + 1) * 127.5).reshape(-1, 3) for im in imgs]

# ── 2. MASt3R → COLMAP ──────────────────────────────────────────────────────
s, R, t = umeyama(c2w[:, :3, 3], cc)
res = np.linalg.norm((s * (R @ c2w[:, :3, 3].T)).T + t - cc, axis=1)
rel = np.sqrt((res ** 2).mean()) / np.linalg.norm(cc - cc.mean(0), axis=1).mean()
print(f"MASt3R→COLMAP カメラ中心の相対RMSE {rel*100:.2f}%（軌道半径比）")
P = [(s * (R @ np.asarray(p).reshape(-1, 3).T)).T + t for p in pts3d]
C = [np.asarray(c).reshape(-1) for c in confs]

# ── 3. 細線の定量（GT座標） ────────────────────────────────────────────────
al = align_colmap_to_gt(src, rec)
span, wire_r, _ = wire_span(src)
split = load_wire_bar_splitter(src)
tot = near = 0
from_wire = []            # 電線マスク画素から出た点の電線表面距離
wire_pix = 0; all_pix = 0
for i, im in enumerate(ims):
    ok = C[i] > args.min_conf
    X = apply(al, P[i])
    d = polyline_distance(X, span, max_exact=0.5) - wire_r
    tot += int(ok.sum()); near += int((ok & (d < 0.03)).sum())
    stem = Path(im.name).stem
    m = cv2.imread(str(src / "gt" / "mask_thin" / f"{stem.split('_')[1]}.png"), 0) > 127
    sp = split(stem, m) if split else None
    if sp is None:
        continue
    wm = cv2.resize(sp[0].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
    wire_pix += int(wm.sum()); all_pix += wm.size
    from_wire.append(d[wm & ok])
fw = np.concatenate(from_wire) if from_wire else np.zeros(0)
stats = {"n_images": len(files), "n_pairs": len(pairs), "cam_rel_rmse": float(rel), "align_gt_rmse_m": al["rmse"],
         "n_dense_conf": tot, "n_near_wire_3cm": near,
         "near_wire_share_pct": 100 * near / max(tot, 1),
         "wire_pixel_share_pct": 100 * wire_pix / max(all_pix, 1),
         "deficiency_wire": (wire_pix / max(all_pix, 1)) / max(near / max(tot, 1), 1e-12),
         "wire_mask_points": int(fw.size),
         "wire_mask_points_within_3cm_pct": float(100 * (fw < 0.03).mean()) if fw.size else None,
         "wire_mask_points_within_10cm_pct": float(100 * (fw < 0.10).mean()) if fw.size else None,
         "wire_mask_points_median_dist_m": float(np.median(fw)) if fw.size else None}
print(json.dumps(stats, indent=1, ensure_ascii=False))

# ── 4. 実験dirを作る（画像・GT はハードリンク、sparse/0 は COLMAP の姿勢＋MASt3R 点群）──────
for sub in ("input", "images", "gt"):
    if not (dst / sub).exists():
        subprocess.run(["cp", "-al", str(src / sub), str(dst / sub)], check=True)
sp0 = dst / "sparse" / "0"; sp0.mkdir(parents=True, exist_ok=True)
for f in ("cameras.bin", "images.bin", "cameras.txt", "images.txt"):
    if (src / "sparse" / "0" / f).exists():
        shutil.copy(src / "sparse" / "0" / f, sp0 / f)
allP = np.concatenate([p[c > args.min_conf] for p, c in zip(P, C)])
allC = np.concatenate([np.asarray(c3)[c > args.min_conf] for c3, c in zip(rgb, C)])
rng = np.random.default_rng(0)
sel = rng.choice(len(allP), size=min(args.n_init, len(allP)), replace=False)
from scene.dataset_readers import storePly  # noqa: E402
storePly(str(sp0 / "points3D.ply"), allP[sel], np.clip(allC[sel], 0, 255).astype(np.uint8))
# 3DGS は points3D.ply があればそれを読む。bin/txt は置かない（元の SfM 点が混ざらないように）
stats["n_init_points"] = int(len(sel))
json.dump(stats, open(dst / "gt" / "mast3r_analysis.json.tmp", "w"), indent=1, ensure_ascii=False)
os.replace(dst / "gt" / "mast3r_analysis.json.tmp", dst / "gt" / "mast3r_analysis.json")
print(f"実験dir: {dst}（初期点 {len(sel)} 点）")

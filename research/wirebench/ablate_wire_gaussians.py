# 残差分解の介入実験：学習済みモデルの「電線スパン近傍ガウシアン」を編集して再レンダ用plyを作る
#   prune_transparent: スパン3cm以内かつ α<しきい値 の半透明ガウシアンを削除
#                      →消して細線PSNRが変わらなければ「半透明残留分」は残差に寄与していない
#   opacify:           同ガウシアンの不透明度を0.99に引き上げ
#                      →上げて改善すれば「透明のまま」が残差の主因、悪化すればブレンドとして機能していた証拠
# 出力: <exp>/<model_dir>_ab_<mode>/point_cloud/iteration_<it>/point_cloud.ply（cfg_args等もコピー）
# 使い方: python3 ablate_wire_gaussians.py <実験dir> [iteration] [モデル出力dir名] --mode prune_transparent
#   その後は通常どおり render→eval_wires（モデルdir名 <model_dir>_ab_<mode> を指定）

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (sc ** 2).sum() * len(src)
    return s, R, mu_d - s * R @ mu_s


parser = argparse.ArgumentParser()
parser.add_argument("exp")
parser.add_argument("iteration", nargs="?", default="7000")
parser.add_argument("model_dir", nargs="?", default="output_mip")
parser.add_argument("--mode", choices=["prune_transparent", "opacify"], required=True)
parser.add_argument("--alpha-th", type=float, default=0.5,
                    help="半透明と見なすしきい値（既定0.5）")
parser.add_argument("--span-th", type=float, default=0.03,
                    help="電線スパン表面からの距離しきい値[m]（既定0.03）")
args = parser.parse_args()

exp = Path(args.exp)
src_model = exp / args.model_dir
ply = PlyData.read(str(src_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"))
v = ply["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))

# スパン近傍の判定（analyze_gaussians.py と同一の定義）
rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
gt = json.load(open(exp / "gt" / "poses.json"))
cc, gc = [], []
for img in rec.images.values():
    stem = Path(img.name).stem
    if stem in gt["frames"]:
        cc.append(img.projection_center())
        gc.append(np.array(gt["frames"][stem])[:3, 3])
s, R, t = umeyama(np.array(cc), np.array(gc))
xyz_w = (s * (R @ xyz.T)).T + t
sp = json.load(open(exp / "gt" / "scene_params.json"))
wp = np.load(exp / "gt" / "wire_points.npz")
wire_only = wp["points"][wp["labels"] == 1]
far = np.min(np.linalg.norm(wire_only[:, None, :2] - np.array(sp["pole_positions"])[None], axis=2), axis=1) > 0.3
surf = cKDTree(wire_only[far]).query(xyz_w)[0] - sp["wire_radius"]
target = (surf < args.span_th) & (alpha < args.alpha_th)
print(f"対象: スパン{args.span_th*100:.0f}cm以内かつ α<{args.alpha_th} → {target.sum()}個 / 全{len(xyz)}個")

data = v.data.copy()
if args.mode == "prune_transparent":
    data = data[~target]
    print(f"削除後: {len(data)}個")
else:
    logit99 = float(np.log(0.99 / 0.01))
    data["opacity"][target] = logit99
    print(f"不透明化: {target.sum()}個の opacity を 0.99 に設定")

dst_model = exp / f"{args.model_dir}_ab_{'prune' if args.mode == 'prune_transparent' else 'opacify'}"
out_ply = dst_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
out_ply.parent.mkdir(parents=True, exist_ok=True)
PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(out_ply))
for f in ["cfg_args", "cameras.json"]:
    if (src_model / f).exists():
        shutil.copy(src_model / f, dst_model / f)
print(f"出力: {dst_model}（render→eval はモデルdir名 {dst_model.name} で通常どおり）")

# 残差分解の介入実験：学習済みモデルの「電線スパン近傍ガウシアン」を編集して再レンダ用plyを作る
#   prune_transparent: スパン3cm以内かつ α<しきい値 の半透明ガウシアンを削除
#                      →消して細線PSNRが変わらなければ「半透明残留分」は残差に寄与していない
#   opacify:           同ガウシアンの不透明度を0.99に引き上げ
#                      →上げて改善すれば「透明のまま」が残差の主因、悪化すればブレンドとして機能していた証拠
# 出力: <exp>/<model_dir>_ab_<mode>/point_cloud/iteration_<it>/point_cloud.ply（cfg_args等もコピー）
# 使い方: python3 ablate_wire_gaussians.py <実験dir> [iteration] [モデル出力dir名] --mode prune_transparent
#   その後は通常どおり render→eval_wires（モデルdir名 <model_dir>_ab_<mode> を指定）
# 2026-09-24（監査R-06/R-13）：半透明の判定を Mip の 3Dフィルタ込み「実効」不透明度に変更（--alpha-kind raw で旧挙動）。
#   近傍判定は中心線への線分距離（旧：13.7cm間隔の点への最近傍＝電線上ガウシアンの約4割を取りこぼしていた）。
#   位置合わせ RMSE>2cm では止める。opacify は ply の生の opacity を 0.99 にするので、Mip では実効値は
#   0.99×フィルタ係数（<0.99）になる点に注意（出力に実効値の分布を表示する）。
#   §9i の旧結果は「生の値＋点最近傍」で群を切ったもの。

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbgeom import align_colmap_to_gt, apply, wire_span, polyline_distance, gaussian_params  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("exp")
parser.add_argument("iteration", nargs="?", default="7000")
parser.add_argument("model_dir", nargs="?", default="output_mip")
parser.add_argument("--mode", choices=["prune_transparent", "opacify"], required=True)
parser.add_argument("--alpha-th", type=float, default=0.5,
                    help="半透明と見なすしきい値（既定0.5）")
parser.add_argument("--alpha-kind", choices=["effective", "raw"], default="effective",
                    help="半透明判定に使う不透明度（既定 effective＝Mipの3Dフィルタ込み。raw は旧挙動）")
parser.add_argument("--near-def", choices=["polyline", "point"], default="polyline",
                    help="電線近傍の距離定義（既定 polyline＝中心線への線分距離。point は旧挙動）")
parser.add_argument("--span-th", type=float, default=0.03,
                    help="電線スパン表面からの距離しきい値[m]（既定0.03）")
args = parser.parse_args()

exp = Path(args.exp)
src_model = exp / args.model_dir
ply = PlyData.read(str(src_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"))
v = ply["vertex"]
xyz, alpha_raw, _, alpha_eff, _, has_filter = gaussian_params(v)
alpha = alpha_eff if args.alpha_kind == "effective" else alpha_raw

# スパン近傍の判定（analyze_gaussians.py と同一の定義）
rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
al = align_colmap_to_gt(exp, rec)
assert al["ok"], f"位置合わせが粗すぎる（RMSE {al['rmse']:.3f}m）"
xyz_w = apply(al, xyz)
span, wire_r, _ = wire_span(exp)
if args.near_def == "polyline":
    surf = polyline_distance(xyz_w, span, max_exact=0.5) - wire_r
else:
    surf = cKDTree(span).query(xyz_w)[0] - wire_r
target = (surf < args.span_th) & (alpha < args.alpha_th)
print(f"対象: スパン{args.span_th*100:.0f}cm以内（{args.near_def}）かつ {args.alpha_kind} α<{args.alpha_th} "
      f"→ {target.sum()}個 / 全{len(xyz)}個" + ("（Mip：3Dフィルタあり）" if has_filter else ""))

data = v.data.copy()
if args.mode == "prune_transparent":
    data = data[~target]
    print(f"削除後: {len(data)}個")
else:
    logit99 = float(np.log(0.99 / 0.01))
    data["opacity"][target] = logit99
    print(f"不透明化: {target.sum()}個の opacity（生の値）を 0.99 に設定")
    if has_filter:
        _, _, _, eff_after, _, _ = gaussian_params(PlyData([PlyElement.describe(data, "vertex")])["vertex"])
        print(f"  Mip の実効不透明度（編集後・対象のみ）中央値 {np.median(eff_after[target]):.3f}")

dst_model = exp / f"{args.model_dir}_ab_{'prune' if args.mode == 'prune_transparent' else 'opacify'}"
out_ply = dst_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
out_ply.parent.mkdir(parents=True, exist_ok=True)
PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(out_ply))
for f in ["cfg_args", "cameras.json"]:
    if (src_model / f).exists():
        shutil.copy(src_model / f, dst_model / f)
print(f"出力: {dst_model}（render→eval はモデルdir名 {dst_model.name} で通常どおり）")

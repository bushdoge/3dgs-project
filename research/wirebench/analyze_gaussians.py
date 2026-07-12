# 学習済みガウシアン(point_cloud.ply)を解剖するスクリプト（残差解剖・霧フローター機構の分析用）
# やること：
#   1. ガウシアン位置を COLMAP座標→GT座標へ Umeyama変換（analyze_sfm.py と同じ手法）
#   2. 電線スパン近傍（表面距離3cm/10cm）のガウシアン数・不透明度・スケールの分布を出す
#      （§9dの「粒はあるのに透明のまま」の定量。不透明度は sigmoid、スケールは exp して相似スケール補正）
#   3. 霧フローター診断：各テスト視点カメラの前方近傍（既定1.5m以内）のガウシアンの
#      「不透明度×断面積」の総和（＝霧の量の代理値）を視点ごとに出す（§9f結論6の機構調査用）
# 使い方: python3 analyze_gaussians.py <実験dir> [iteration] [モデル出力dir名] [--near-cam 1.5]
#   例: python3 analyze_gaussians.py experiments/20260704_wb_r15mm_f80_open 7000 output_mip
# 出力: 標準出力＋ <実験dir>/gt/gaussian_analysis_<model_dir>_<iteration>.json

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData
from scipy.spatial import cKDTree


def fresh(p):
    # cp -al 複製由来のinode共有による巻き添え上書きを防ぐ（eval_wires.py と同じ）
    p = Path(p)
    p.unlink(missing_ok=True)
    return p


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


def q(a, ps=(25, 50, 75)):
    return [round(float(np.percentile(a, p)), 4) for p in ps] if len(a) else [None] * len(ps)


parser = argparse.ArgumentParser()
parser.add_argument("exp", help="実験ディレクトリ")
parser.add_argument("iteration", nargs="?", default="7000")
parser.add_argument("model_dir", nargs="?", default="output")
parser.add_argument("--near-cam", type=float, default=1.5,
                    help="霧診断でカメラ前方この距離[m]以内を数える（既定1.5）")
parser.add_argument("--fog-dump", type=int, default=0,
                    help="寄与上位N個のガウシアンをダンプ（位置・色・不透明度・スパン距離）")
parser.add_argument("--fog-view", default=None,
                    help="ダンプ対象視点（例 frame_0009）。省略時はfog_alpha最大の視点。"
                         "霧の有無の検出はレンダ側指標(R_ren等)で行い、ここでは組成を見るのが正しい使い方")
args = parser.parse_args()

exp = Path(args.exp)
ply_path = exp / args.model_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
v = PlyData.read(str(ply_path))["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))          # raw→sigmoid
scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
# 色: SH0次係数→RGB（0〜1にクリップ）。霧ガウシアンが注入点由来（暗灰色）かの判定に使う
rgb = np.clip(0.5 + 0.28209479 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
print(f"ガウシアン総数: {len(xyz)}（{ply_path}）")

# ── COLMAP→GT 位置合わせ（カメラ中心の対応から推定）────────────────────────────
rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
gt = json.load(open(exp / "gt" / "poses.json"))
colmap_c, gt_c = [], []
for img in rec.images.values():
    stem = Path(img.name).stem
    if stem in gt["frames"]:
        colmap_c.append(img.projection_center())
        gt_c.append(np.array(gt["frames"][stem])[:3, 3])
s, R, t = umeyama(np.array(colmap_c), np.array(gt_c))
rmse = np.sqrt((((s * (R @ np.array(colmap_c).T)).T + t - np.array(gt_c)) ** 2).sum(1).mean())
print(f"位置合わせ RMSE: {rmse*1000:.1f}mm（スケール {s:.4f}）")
assert rmse < 0.02, "位置合わせが粗すぎる（>2cm）"
xyz_w = (s * (R @ xyz.T)).T + t
scales_w = scales * s                      # 相似スケールで実寸[m]へ

# ── 電線スパン近傍の統計（analyze_sfm.py の厳密版と同じ定義）──────────────────────
wp = np.load(exp / "gt" / "wire_points.npz")
wire_only = wp["points"][wp["labels"] == 1]
sp = json.load(open(exp / "gt" / "scene_params.json"))
WIRE_R = sp["wire_radius"]
POLE_XY = np.array(sp["pole_positions"])
far = np.min(np.linalg.norm(wire_only[:, None, :2] - POLE_XY[None], axis=2), axis=1) > 0.3
span = wire_only[far]
surf_dist = cKDTree(span).query(xyz_w)[0] - WIRE_R

result = {"model_dir": args.model_dir, "iteration": int(args.iteration),
          "n_gaussians": len(xyz), "align_rmse_m": round(float(rmse), 4),
          "wire_radius_m": WIRE_R, "span_stats": {}, "near_cam_m": args.near_cam,
          "fog_per_test_view": {}}
print(f"\n[電線スパン近傍のガウシアン]（表面距離。スパンGT {len(span)}点・半径{WIRE_R*1000:.0f}mm補正）")
for name, th in [("3cm", 0.03), ("10cm", 0.10)]:
    m = surf_dist < th
    op, sc_max, sc_min = opacity[m], scales_w[m].max(1), scales_w[m].min(1)
    st = {"count": int(m.sum()),
          "opacity_q25_50_75": q(op),
          "opacity_ge_0.5_count": int((op >= 0.5).sum()),
          "scale_max_axis_mm_q25_50_75": q(sc_max * 1000),
          "anisotropy_q25_50_75": q(sc_max / np.maximum(sc_min, 1e-9))}
    result["span_stats"][name] = st
    print(f"  {name}以内: {st['count']:6d}個  不透明度[q25/50/75]={st['opacity_q25_50_75']}"
          f"  (≥0.5が{st['opacity_ge_0.5_count']}個)  最大軸[mm]={st['scale_max_axis_mm_q25_50_75']}")

# ── 霧診断：テスト視点カメラ前方近傍の「不透明度×断面積」総和 ─────────────────────
# 断面積の代理として上位2軸の積（πは省略。相対比較用）。fog_massが大きい視点は
# カメラ前に半透明の大きいガウシアンが溜まっている＝霧の疑い。
names = sorted(gt["frames"].keys())
test_names = names[::8]
# 霧の指標＝画像中心の視線を近距離で塞ぐ「濃い遮蔽体」の累積アルファ。
# 各ガウシアンの視線への寄与を α_i = 不透明度 × exp(-perp²/2σ²)（σ=最大軸）で近似し、
# α_i > 0.3 の濃い遮蔽体だけで 1-Π(1-α_i) を計算する（微小αの大群では飽和しない）。
# σ=最大軸は楕円の向きを無視した過大評価だが、不透明度の高い遮蔽体に限るので実害が小さい。
print(f"\n[霧診断] 中心視線を塞ぐ濃い遮蔽体（α>0.3・カメラから{args.near_cam}m以内）の累積 fog_alpha")
fog_cache = {}
for name in test_names:
    M = np.array(gt["frames"][name])
    C, fwd = M[:3, 3], -M[:3, 2]                     # Blender/OpenGLは-Zが前方
    rel = xyz_w - C
    t_ray = rel @ fwd                                 # 視線方向の距離
    perp = np.linalg.norm(rel - t_ray[:, None] * fwd[None], axis=1)
    sigma = scales_w.max(1)
    alpha = opacity * np.exp(-0.5 * (perp / np.maximum(sigma, 1e-6)) ** 2)
    near = (t_ray > 0) & (t_ray < args.near_cam) & (alpha > 0.3)
    fog_alpha = float(1 - np.prod(1 - np.clip(alpha[near], 0, 0.999)))
    contrib = alpha[near]
    fog_cache[name] = (near, contrib)
    st = {"count": int(near.sum()), "fog_alpha": round(fog_alpha, 4),
          "alpha_sum": round(float(alpha[near].sum()), 2)}
    result["fog_per_test_view"][name] = st
    print(f"  {name}: n={st['count']:5d}  fog_alpha={st['fog_alpha']:7.4f}  Σα={st['alpha_sum']:8.2f}")

# ── 霧ダンプ：fog_mass最大の視点で寄与上位N個の素性を見る（注入点由来かの判定用）───────
if args.fog_dump > 0:
    worst = args.fog_view or max(result["fog_per_test_view"],
                                 key=lambda n: result["fog_per_test_view"][n]["fog_alpha"])
    near, contrib = fog_cache[worst]
    idx_near = np.where(near)[0]
    order = np.argsort(contrib)[::-1][:args.fog_dump]
    print(f"\n[霧ダンプ] {worst}（fog_alpha最大）の中心視線への寄与上位{len(order)}個:")
    print(f"  {'寄与α':>8s} {'不透明度':>7s} {'最大軸mm':>9s} {'RGB(0-255)':>14s} {'スパン距離m':>10s} {'カメラ距離m':>10s}")
    M = np.array(gt["frames"][worst])
    C = M[:3, 3]
    dump = []
    for k in order:
        i = idx_near[k]
        c255 = (rgb[i] * 255).astype(int)
        row = {"contrib": round(float(contrib[k]), 4),
               "opacity": round(float(opacity[i]), 3),
               "scale_max_mm": round(float(scales_w[i].max() * 1000), 1),
               "rgb": c255.tolist(),
               "span_dist_m": round(float(surf_dist[i] + WIRE_R), 3),
               "cam_dist_m": round(float(np.linalg.norm(xyz_w[i] - C)), 3),
               "pos": [round(float(x), 3) for x in xyz_w[i]]}
        dump.append(row)
        print(f"  {row['contrib']:8.4f} {row['opacity']:7.3f} {row['scale_max_mm']:9.1f} "
              f"{str(row['rgb']):>14s} {row['span_dist_m']:10.3f} {row['cam_dist_m']:10.3f}")
    result["fog_dump_view"] = worst
    result["fog_dump"] = dump

out = fresh(exp / "gt" / f"gaussian_analysis_{args.model_dir}_{args.iteration}.json")
json.dump(result, open(out, "w"), indent=1, ensure_ascii=False)
# 不透明度分布の生値（ヒストグラム図用）: スパン3cm以内のガウシアンの不透明度と最大軸長
m3 = surf_dist < 0.03
np.savez(fresh(exp / "gt" / f"gaussian_analysis_{args.model_dir}_{args.iteration}.npz"),
         opacity_3cm=opacity[m3], scale_max_3cm=scales_w[m3].max(1))
print(f"\n保存: {out}（＋同名.npz: スパン3cm内の不透明度・最大軸の生値）")

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


def fresh(p):
    # 実験dirを cp -al で複製すると gt/ 配下の成果物が元dirとinode共有になり、
    # 上書き保存が元実験の結果を巻き添えで壊す。書く前にunlinkしてリンクを切る。
    p = Path(p)
    p.unlink(missing_ok=True)
    return p


exp = Path(sys.argv[1])
rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
gt = json.load(open(exp / "gt" / "poses.json"))
wp = np.load(exp / "gt" / "wire_points.npz")
wire_pts, wire_labels = wp["points"], wp["labels"]

# ── COLMAP→GT の相似変換（カメラ中心の対応から。wbgeom.align_colmap_to_gt）──────────
# 素朴な Umeyama の RMSE が 2cm を超えたら外れカメラを除いてロバストに再推定する（監査R-07：
# crossing f40 は3台だけ約10mずれており、素朴な推定では RMSE 3.2m で全点の距離が無意味になっていた）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbgeom import align_colmap_to_gt, polyline_distance      # noqa: E402

al = align_colmap_to_gt(exp, rec)
s, R, t = al["s"], al["R"], al["t"]
rmse = al["rmse"]
print(f"登録画像: {al['n_cams']} / {len(gt['frames'])}")
print(f"位置合わせ RMSE: {rmse:.4f} m（スケール {s:.4f}）")
if al["rmse_naive"] > 0.02:
    print(f"[警告] 素朴な位置合わせの RMSE が {al['rmse_naive']:.3f} m。外れカメラ {len(al['outlier_frames'])} 台 "
          f"{al['outlier_frames']} を除いて再推定した（インライア {al['n_inliers']} 台で RMSE {rmse*1000:.1f} mm）")
if not al["ok"]:
    print("[警告] インライアでも RMSE が 2cm を超える＝SfMそのものが破綻。以下の数値は信頼できない")

# ── SfM点群を変換して細線からの距離を測る ──────────────────────────────────────
pts3d = np.array([p.xyz for p in rec.points3D.values()])
pts3d_w = (s * (R @ pts3d.T)).T + t
tree = cKDTree(wire_pts)
dist, idx = tree.query(pts3d_w)

for th_name, th in [("5cm", 0.05), ("10cm", 0.10), ("20cm", 0.20)]:
    near = dist < th
    n_wire = (wire_labels[idx[near]] == 1).sum()
    n_bar = (wire_labels[idx[near]] == 2).sum()
    print(f"細線から{th_name}以内のSfM点: {near.sum():5d} / {len(pts3d)} "
          f"({near.sum()/len(pts3d)*100:.3f}%)  [電線 {n_wire} / フェンス棒 {n_bar}]")

# ── 厳密版：電線スパン部（電柱から0.3m以上離れた区間）だけで数える ────────────────
# 緩い閾値だと電柱・地面・レール由来の近傍点を拾ってしまうため、
# 電柱の影響を受けない純粋な電線区間だけで測るのが本命の指標。
# 注意: GTサンプルは電線の中心線上、SfM点が付くのは電線の表面。
#       太い電線ほど中心線から遠くなるので、距離から線半径を引いた「表面距離」で閾値判定する。
sp_path = exp / "gt" / "scene_params.json"
sp = json.load(open(sp_path)) if sp_path.exists() else {}
WIRE_R = sp.get("wire_radius", 0.015)
# 電柱位置は scene_params.json（make_scene.py が書く）から読む。--layout で本数・位置が変わるため。
# 旧実験の scene_params.json には pole_positions が無いことがあるので base の値をfallbackにする。
POLE_XY = np.array(sp.get("pole_positions", [[-4.0, -1.8], [4.0, -1.2]]), dtype=float)[:, :2]
wire_only = wire_pts[wire_labels == 1]
far_from_pole = np.min(np.linalg.norm(wire_only[:, None, :2] - POLE_XY[None], axis=2), axis=1) > 0.3
span = wire_only[far_from_pole]

# 距離は「GTサンプル点への最近傍」ではなく「中心線ポリラインへの線分距離」で測る。
# GTサンプルは13.7cm間隔と粗く、点への最近傍距離だとサンプル間の中央にあるSfM点を
# 取りこぼす（幾何計算で捕捉率約62%＝厳密カウントが系統的に約4割過小。監査C2）。
# wire_points は電線ごとに順に並んでいるので、連続点間の距離が飛ぶ所で線を分割する。
# 線分距離の実装は wbgeom.polyline_distance（analyze_gaussians / ablate_wire_gaussians と共通）
sdist_pt, _ = cKDTree(span).query(pts3d_w)            # 旧定義（比較用に残す）
sdist = polyline_distance(pts3d_w, span)
surf_dist = sdist - WIRE_R                            # 電線表面からの距離
seg_len = np.median(np.linalg.norm(np.diff(span, axis=0), axis=1))
print(f"\n[厳密] 電線スパン部GTサンプル: {len(span)}点（サンプル間隔 中央値{seg_len*100:.1f}cm）"
      f"／線分距離で判定・線半径 {WIRE_R*1000:.0f}mm を補正")
strict = {}
for th in (0.03, 0.05):
    n = int((surf_dist < th).sum())
    n_old = int(((sdist_pt - WIRE_R) < th).sum())
    strict[f"{int(th*100)}cm"] = n
    print(f"[厳密] 電線スパン表面から{th*100:.0f}cm以内のSfM点: {n} / {len(pts3d)} "
          f"({n/len(pts3d)*100:.4f}%)  [旧・点最近傍だと {n_old}]")

# ── 比較用：画素占有率（GTマスクの平均白率）──────────────────────────────────
# 混合マスク全体に加え、電線スパン部だけの占有率も測る。欠乏倍率の分子と分母で
# 母集団（混合 vs 電線のみ）が食い違っていたため（監査C1）。
# 分類は eval_wires.py と同じ「GT点を投影して最近傍ラベル」方式を共有する。
from eval_wires import load_wire_bar_splitter          # noqa: E402

mask_dir = exp / "gt" / "mask_thin"
# 厳密カウントと母集団を揃えるため、「電線スパン部」だけを1類として分類させる
# （電柱近傍0.3m以内の電線とフェンス棒は2類にまとめる）。
span_labels = np.full(len(wire_pts), 2, dtype=np.int32)
is_wire = wire_labels == 1
span_labels[np.where(is_wire)[0][far_from_pole]] = 1
splitter = load_wire_bar_splitter(exp, labels=span_labels)
fracs, wfracs = [], []
for f in sorted(mask_dir.glob("*.png"))[::8]:
    m = cv2.imread(str(f), 0)
    mb = m > 127
    fracs.append(mb.mean())
    if splitter is not None:
        r = splitter(f"frame_{f.stem}" if not f.stem.startswith("frame_") else f.stem, mb)
        if r is not None:
            wfracs.append(r[0].mean())
pix_share = np.mean(fracs) * 100
wire_pix_share = float(np.mean(wfracs) * 100) if wfracs else float("nan")
print(f"\n細線（電線＋棒）の平均画素占有率: {pix_share:.3f}%")
print(f"電線スパン部のみの平均画素占有率: {wire_pix_share:.3f}%（{len(wfracs)}視点）")

# ── 欠乏倍率 ──────────────────────────────────────────────────────────────────
# 主指標（2026-09-07 改定）＝ **電線のみで分子と分母を揃えた**もの:
#   分子 = 電線スパン部の画素占有率 / 分母 = 電線スパン表面3cm以内のSfM点の割合。
# 旧指標（混合マスクの画素占有率 ÷ 混合10cm以内の点割合）は互換のため併記する。
wire_pt_share = strict["3cm"] / len(pts3d) * 100
# 点が0個だと倍率は発散するので、ポアソン95%上限（n=0 のとき3.0点）で**下界**として報告する。
# 高解像度でも電線上の点は2〜11個しかなく、倍率の点推定は計数ノイズ支配であることに注意（§9ah）。
if strict["3cm"] == 0:
    deficiency = wire_pix_share / (3.0 / len(pts3d) * 100)
    defic_note = f"≧{deficiency:.0f}倍（電線上の点0個。ポアソン95%上限3点で計算した下界）"
else:
    deficiency = wire_pix_share / wire_pt_share
    defic_note = f"{deficiency:.0f}倍の欠乏"
mixed_pt_share = (dist < 0.10).sum() / len(pts3d) * 100
mixed_deficiency = pix_share / max(mixed_pt_share, 1e-9)

json.dump({"n_registered": al["n_cams"], "n_points3d": len(pts3d),
           "align_rmse_m": float(rmse), "align_rmse_naive_m": al["rmse_naive"],
           "align_ok": bool(al["ok"]), "outlier_frames": al["outlier_frames"], "wire_span_points": strict,
           "thin_pixel_share_pct": float(pix_share),
           "wire_pixel_share_pct": wire_pix_share,
           "wire_point_share_pct": float(wire_pt_share),
           "deficiency_wire_only": float(deficiency),
           "deficiency_is_lower_bound": strict["3cm"] == 0,
           "deficiency_mixed_legacy": float(mixed_deficiency)},
          open(fresh(exp / "gt" / "sfm_analysis.json"), "w"), indent=1)
print(f"\n→ [主指標・電線のみ] 画素の{wire_pix_share:.3f}%を占める電線スパンに、"
      f"SfM点は{strict['3cm']}点 / {len(pts3d)}点（{wire_pt_share:.4f}%）しか無い（**{defic_note}**）")
print(f"→ [旧指標・混合] 画素{pix_share:.2f}% vs 点{mixed_pt_share:.3f}%"
      f"（{mixed_deficiency:.0f}倍。分子=混合・分母=混合10cm以内で母集団が緩い）")

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
out = fresh(exp / "gt" / "sfm_points_overlay.png")
cv2.imwrite(str(out), img)
print(f"\n可視化を保存: {out}（赤=SfM点。細線上に点が無いことを確認）")

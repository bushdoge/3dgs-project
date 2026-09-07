# P2-0b診断：裾（誤差上位断面）の3D対応分析（設計図: memo/design_20260717_phase2.md の P2-0b）
# 問い：悪い線分は「同じ3D区間が複数視点で悪い」（幾何起因）のか「視点ごとに別の場所」（見え方起因）なのか。
# 方法：tail_map.py と同じ断面抽出を行い、各断面をGTワイヤ点（wire_points.npz）の投影最近傍
#       （GT深度で遮蔽チェック済みの可視点のみ）に対応付け、
#       (1) 視点半分ずつ（偶数/奇数）で集計した3D点ごとの誤差のSpearman相関（高い＝幾何起因）
#       (2) 各視点の上位5%断面が触る3D点集合の視点間Jaccard重なり（チャンス水準と比較）
#       (3) 視点の誤差シェアとカメラ幾何（線との視線角・距離）の関係
# 使い方: python3 tail_corr3d.py <実験dir> [iteration] [モデル出力dir名]
# 出力: 標準出力＋ <実験dir>/gt/tail_corr3d_<model>_<iter>.json
# 依存: pip install OpenEXR（GT深度の読み込み。2026-07-17にシステムpython3へ導入済み）

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath


def read_depth(path):
    f = OpenEXR.InputFile(str(path))
    h = f.header()
    ch = list(h["channels"].keys())[0]
    dw = h["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    return np.frombuffer(f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT)),
                         np.float32).reshape(H, W)


def dip_centroid(prof, win):
    bg = np.median(np.concatenate([prof[:3], prof[-3:]]))
    dip = bg - prof
    if dip.max() < 8:
        return None
    return True  # 本スクリプトでは測定可否のみ使う


parser = argparse.ArgumentParser()
parser.add_argument("exp")
parser.add_argument("iteration", nargs="?", default="7000")
parser.add_argument("model_dir", nargs="?", default="output_mip")
parser.add_argument("--win", type=int, default=14)
parser.add_argument("--assoc-px", type=float, default=8.0)
args = parser.parse_args()

exp = Path(args.exp)
rdir = exp / args.model_dir / "test" / f"ours_{args.iteration}"
names = test_view_names(exp)
win = args.win

poses = json.load(open(exp / "gt" / "poses.json"))
fx, fy, cx0, cy0 = poses["fx"], poses["fy"], poses["cx"], poses["cy"]
z = np.load(exp / "gt" / "wire_points.npz")
P, labels = z["points"], z["labels"]
# 接線（同ラベル内の隣接点差分）
T = np.zeros_like(P)
for lb in np.unique(labels):
    idx = np.where(labels == lb)[0]
    t = np.gradient(P[idx], axis=0)
    T[idx] = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


def project(name):
    """GTワイヤ点を視点nameへ投影し、GT深度で遮蔽点を除外する。
    返り値: (u,v) [N,2], カメラ距離 [N], 視線と接線のなす角cos [N], 可視フラグ [N]"""
    C = np.array(poses["frames"][name])          # cam2world (OpenGL: -Z視線, +Y上)
    W2C = np.linalg.inv(C)
    Xc = (W2C[:3, :3] @ P.T + W2C[:3, 3:4]).T
    zc = -Xc[:, 2]                                # 前方が正
    u = cx0 + fx * Xc[:, 0] / np.maximum(zc, 1e-6)
    v = cy0 - fy * Xc[:, 1] / np.maximum(zc, 1e-6)
    cam_pos = C[:3, 3]
    ray = P - cam_pos
    dist = np.linalg.norm(ray, axis=1)
    cosang = np.abs((ray / dist[:, None] * T).sum(1))
    ok = zc > 0.1
    # 遮蔽チェック: 投影画素の3×3近傍のGT深度最小値が点の深度と整合するか
    # （細線はサブピクセルなので深度画素は背景を指しうる→近傍最小と10%+5cmの緩い許容で判定）
    dep = read_depth(exp / "gt" / "depth" / f"{name.split('_')[1]}.exr")
    H, W = dep.shape
    dmin = cv2.erode(dep, np.ones((3, 3), np.float32))
    ui, vi = np.clip(u, 0, W - 1).astype(int), np.clip(v, 0, H - 1).astype(int)
    near = dmin[vi, ui]
    # Blender Zパスの規約（視線距離 or カメラ面距離）は実データで判定
    ref = dist if np.median(np.abs(near - dist)) < np.median(np.abs(near - zc)) else zc
    visible = ok & (near >= ref - np.maximum(0.10 * ref, 0.05))
    return np.stack([u, v], 1), dist, cosang, visible


# 断面抽出＋3D対応付け
recs = []      # (view, point_idx, err)
view_geom = {}
for i, name in enumerate(names):
    ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
    gt = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
    mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{name.split('_')[1]}.png"), 0)
    if ren is None or gt is None or mask is None:
        continue
    mb = mask > 127
    gray_r = cv2.cvtColor(ren, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
    gray_g = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
    uv, dist, cosang, visible = project(name)
    h, w = mb.shape
    # 投影の妥当性: 画面内の可視GT点がマスク近傍(3px膨張)に乗る割合
    inside = visible & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    md = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 127
    hit = md[np.clip(uv[inside, 1].astype(int), 0, h - 1), np.clip(uv[inside, 0].astype(int), 0, w - 1)]
    view_geom[i] = {"name": name, "proj_on_mask": round(float(hit.mean()), 3),
                    "n_inside": int(inside.sum()),
                    "mean_dist": round(float(dist[inside].mean()), 2),
                    "mean_abs_cos_tangent": round(float(cosang[inside].mean()), 3),
                    "err_total": 0.0}
    vis_idx = np.where(inside)[0]
    for cxp in range(0, w, 4):
        col = mb[:, cxp]
        if not col.any():
            continue
        ys = np.where(col)[0]
        for run in np.split(ys, np.where(np.diff(ys) > 1)[0] + 1):
            if len(run) > win:
                continue
            cyp = int(run.mean())
            y0, y1 = cyp - win, cyp + win + 1
            if y0 < 0 or y1 > h:
                continue
            e0 = float(((gray_r[y0:y1, cxp] - gray_g[y0:y1, cxp]) ** 2).sum())
            d2 = (uv[vis_idx, 0] - cxp) ** 2 + (uv[vis_idx, 1] - cyp) ** 2
            j = int(np.argmin(d2))
            if d2[j] <= args.assoc_px ** 2:
                recs.append((i, int(vis_idx[j]), e0))
            view_geom[i]["err_total"] += e0

views = np.array([r[0] for r in recs])
pts = np.array([r[1] for r in recs])
errs = np.array([r[2] for r in recs])
uviews = sorted(set(views.tolist()))
print(f"断面→3D点の対応付け: {len(recs)}断面 / 対応3D点 {len(set(pts.tolist()))}個")
for v in uviews:
    g = view_geom[v]
    print(f"  view {v}: 投影のマスク乗り率 {g['proj_on_mask']*100:.0f}%（要>85%: 投影の健全性チェック）")

# (1) 偶数視点 vs 奇数視点で3D点ごとの平均誤差 → Spearman相関
def per_point_mean(sel):
    s = {}
    for vv, pp, ee in zip(views[sel], pts[sel], errs[sel]):
        s.setdefault(pp, []).append(ee)
    return {k: np.mean(v) for k, v in s.items()}

A = per_point_mean(np.isin(views, uviews[::2]))
B = per_point_mean(np.isin(views, uviews[1::2]))
common = sorted(set(A) & set(B))
from scipy.stats import spearmanr
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent)); from testviews import test_view_names
rho, pval = spearmanr([A[k] for k in common], [B[k] for k in common])
print(f"[1] 3D点ごとの誤差の視点半区分間Spearman相関: rho={rho:.3f} (p={pval:.1e}, n={len(common)})")

# (2) 視点ごとの上位5%断面が触る3D点集合のJaccard重なり（対チャンス）
tops = {}
for v in uviews:
    m = views == v
    if m.sum() < 20:
        continue
    th = np.quantile(errs[m], 0.95)
    tops[v] = set(pts[m][errs[m] >= th].tolist())
pairs, jac = 0, []
rng = np.random.default_rng(0)
jac_rand = []
for a in tops:
    for b in tops:
        if a >= b:
            continue
        va, vb = tops[a], tops[b]
        if not va or not vb:
            continue
        jac.append(len(va & vb) / len(va | vb))
        # チャンス: 同サイズをその視点の全対応点からランダム抽出
        pa = np.unique(pts[views == a]); pb = np.unique(pts[views == b])
        ra = set(rng.choice(pa, min(len(va), len(pa)), replace=False).tolist())
        rb = set(rng.choice(pb, min(len(vb), len(pb)), replace=False).tolist())
        jac_rand.append(len(ra & rb) / max(len(ra | rb), 1))
print(f"[2] 上位5%断面が触る3D点集合の視点間Jaccard: 観測 {np.mean(jac):.3f} vs チャンス {np.mean(jac_rand):.3f} ({len(jac)}ペア)")

# (3) 視点誤差シェアとカメラ幾何
tot = sum(g["err_total"] for g in view_geom.values())
rows = []
for v in uviews:
    g = view_geom[v]
    rows.append((v, g["err_total"] / tot, g["mean_dist"], g["mean_abs_cos_tangent"]))
share = np.array([r[1] for r in rows])
rd, _ = spearmanr(share, [r[2] for r in rows])
rc, _ = spearmanr(share, [r[3] for r in rows])
print(f"[3] 視点誤差シェアとの相関: 距離 rho={rd:.2f} / |cos(視線,接線)| rho={rc:.2f}")
for r in sorted(rows, key=lambda x: -x[1]):
    print(f"    view {r[0]}: share {r[1]*100:4.1f}% dist {r[2]:.1f} |cos| {r[3]:.2f}")

res = {"model_dir": args.model_dir, "iteration": int(args.iteration),
       "n_sections": len(recs), "n_points": len(set(pts.tolist())),
       "proj_on_mask_per_view": {v: view_geom[v]["proj_on_mask"] for v in uviews},
       "spearman_halfsplit_rho": round(float(rho), 3), "spearman_halfsplit_p": float(pval),
       "top5_jaccard_obs": round(float(np.mean(jac)), 3),
       "top5_jaccard_chance": round(float(np.mean(jac_rand)), 3),
       "view_share_vs_dist_rho": round(float(rd), 2),
       "view_share_vs_costangent_rho": round(float(rc), 2),
       "per_view": [{"view": r[0], "err_share": round(r[1], 3), "mean_dist": r[2],
                     "mean_abs_cos_tangent": r[3]} for r in rows]}
out = exp / "gt" / f"tail_corr3d_{args.model_dir}_{args.iteration}.json"
out.unlink(missing_ok=True)
json.dump(res, open(out, "w"), indent=1)
print(f"保存: {out}")

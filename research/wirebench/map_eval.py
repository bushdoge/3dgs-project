# 3DGS を「地図」として見たときの評価（研究メモ §9at）。3DGS を地図にした経路計画（Splat-Nav, arXiv 2403.02751）と同じく、
# 各ガウシアンの γ 信頼楕円体（既定 γ=0.2、χ²₃(0.2)≈1.005＝ほぼ1σ）の和集合を「物がある空間」とみなし、2方向の誤りを測る。
#   見逃し：電線の中心線を 1cm 間隔で区切り、各点に楕円体が届いているか（(x−μ)ᵀ(Σ+R²I)⁻¹(x−μ) ≤ χ²₃(γ)、R＝電線半径。
#           楕円体と電線の太さのミンコフスキー和の近似）。被覆率と、覆われていない区間（＝計画器が通り抜けうる隙間）の長さを出す。
#   誤検知：GT 深度で「何も無い」と確かめられた点（どれかの GT 視点で、周囲 --free-margin の窓の GT 深度より --free-margin 以上手前）が
#           楕円体の中に入っている割合（κ=0）と、半径 --kappa の球の機体が楕円体と交わる割合（Σ+κ²I の近似）。領域は (a) 電線の周り 0.15〜1.0m の殻（電線を越える経路が通る所）(b) シーン全体の箱。
# 画素の誤差（PSNR）は、面積の小さい物体の欠落にも、背景と同じ色の偽の障害物にも鈍いので、地図用途にはこの2本で評価する。
# 注意：LOD-GS は描画時に係数（coefs）で不透明度・スケールを変えるが、ここでは ply の生の値を使う。2DGS は厚み0の円盤として扱う。
#       Mip 系（filter_3D を持つ ply）は 3D フィルタ込みの実効スケール・不透明度。
# 使い方: python3 map_eval.py <exp> <model_dir名> [...] [--iter 30000] [--gamma 0.2] [--out <csv>]
#   → 既定で <exp>/gt/map_eval.csv（1行＝1モデル。--append で追記）
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycolmap
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbgeom import align_colmap_to_gt, apply, wire_span    # noqa: E402
from probe_collision import quat_to_R                       # noqa: E402
from scene_stats import read_depth                          # noqa: E402

STEP = 0.01          # 電線のサンプル間隔[m]
GAP_DRONE = 0.3      # 「機体が通れる隙間」の目安[m]（小型ドローンの直径程度）
EPS2 = 1e-8          # 2DGS（厚み0）の共分散を正則にする[m²]


def load_gaussians(ply, al):
    """ply → GT 座標の中心・共分散・実効不透明度"""
    v = PlyData.read(str(ply))["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3) if f"scale_{i}" in names], 1).astype(np.float64))
    if sc.shape[1] == 2:                                        # 2DGS：法線方向の厚みは0
        sc = np.concatenate([sc, np.zeros((len(sc), 1))], 1)
    if "filter_3D" in names:                                    # Mip 系：wbgeom.gaussian_params と同じ
        f2 = np.asarray(v["filter_3D"], np.float64)[:, None] ** 2
        op = op * np.sqrt(np.prod(sc ** 2 / (sc ** 2 + f2), axis=1))
        sc = np.sqrt(sc ** 2 + f2)
    Rq = quat_to_R(np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64))
    Rw = al["R"][None] @ Rq
    sw = sc * al["s"]
    cov = (Rw * sw[:, None, :] ** 2) @ Rw.transpose(0, 2, 1) + EPS2 * np.eye(3)
    return apply(al, xyz), cov, op, sw.max(1)


def inside_any(Q, mu, cov, smax, thr, infl2=0.0, chunk=400000):
    """Q の各点が、どれかの楕円体（Σ + infl2·I）の閾値内にあるか。中心の近傍探索で候補対を絞る"""
    hit = np.zeros(len(Q), bool)
    tree = cKDTree(Q)
    reach = np.sqrt(thr * (smax ** 2 + infl2)) + 1e-4
    lo, hi = Q.min(0), Q.max(0)
    cand = np.nonzero(((mu > lo - reach[:, None]) & (mu < hi + reach[:, None])).all(1))[0]
    for s in range(0, len(cand), chunk):
        g = cand[s:s + chunk]
        nb = tree.query_ball_point(mu[g], reach[g], workers=-1)
        cnt = np.fromiter((len(x) for x in nb), int, len(nb))
        if cnt.sum() == 0:
            continue
        gi = np.repeat(g, cnt)
        qi = np.fromiter((j for x in nb for j in x), int, cnt.sum())
        for t in range(0, len(gi), 2_000_000):
            a, b = gi[t:t + 2_000_000], qi[t:t + 2_000_000]
            inv = np.linalg.inv(cov[a] + infl2 * np.eye(3))
            d = Q[b] - mu[a]
            m2 = np.einsum("ni,nij,nj->n", d, inv, d)
            hit[b[m2 <= thr]] = True
    return hit


def wire_polylines(exp):
    """電線（電柱から 0.3m 以上離れたスパン部）を 1cm 間隔に補間した折れ線のリスト"""
    pts, R, _ = wire_span(exp)
    breaks = np.where(np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.5)[0] + 1
    out = []
    for seg in np.split(pts, breaks):
        if len(seg) < 2:
            continue
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        s = np.arange(0, cum[-1], STEP)
        out.append(np.stack([np.interp(s, cum, seg[:, k]) for k in range(3)], 1))
    return out, R


def free_points(exp, wires, n, margin, rng):
    """GT 深度で空いていると確かめられた点。(a) 電線の周りの殻 (b) シーンの箱"""
    W = np.concatenate(wires)
    k = rng.integers(0, len(W), n)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    shell = W[k] + dirs * rng.uniform(0.15, 1.0, n)[:, None]
    box = np.c_[rng.uniform(-6, 6, n), rng.uniform(-6, 6, n), rng.uniform(0.3, 6.0, n)]
    poses = json.load(open(exp / "gt" / "poses.json"))
    f, Wd, Hd = poses["fx"], poses["width"], poses["height"]
    radii = [2, 4, 8, 16, 32, 64]
    res = {}
    for name, P in (("shell", shell), ("box", box)):
        free = np.zeros(len(P), bool)
        for key, M in poses["frames"].items():
            dp = exp / "gt" / "depth" / f"{key.split('_')[-1]}.exr"
            if not dp.exists():
                continue
            D = read_depth(dp).astype(np.float64)
            D[~np.isfinite(D) | (D > 1e4)] = 1e6
            M = np.array(M)
            pc = (M[:3, :3].T @ (P - M[:3, 3]).T).T
            dep = -pc[:, 2]
            u = f * pc[:, 0] / np.maximum(dep, 1e-6) + Wd / 2
            v = Hd / 2 - f * pc[:, 1] / np.maximum(dep, 1e-6)
            r_need = np.ceil(f * margin / np.maximum(dep, 1e-6))
            ok = (dep > margin) & (u >= r_need) & (u < Wd - r_need) & (v >= r_need) & (v < Hd - r_need) & (r_need <= radii[-1])
            for i, r in enumerate(radii):
                sel = ok & (r_need <= r) & ((r_need > radii[i - 1]) if i else True)
                if not sel.any():
                    continue
                Dm = cv2.erode(D, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)))   # 窓内の最小深度
                ui, vi = u[sel].astype(int), v[sel].astype(int)
                free[np.nonzero(sel)[0][dep[sel] < Dm[vi, ui] - margin]] = True
        res[name] = P[free]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--n-free", type=int, default=40000, help="自由空間のサンプル数（領域ごと、空き判定の前）")
    ap.add_argument("--free-margin", type=float, default=0.1, help="空き判定の余裕[m]（深度方向・横方向とも）")
    ap.add_argument("--kappa", type=float, default=0.05, help="機体半径[m]。誤検知を機体（球）との衝突でも測る（2DGS の厚み0の円盤を公平に扱うため）。--free-margin 以下にする")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    thr = chi2.ppf(a.gamma, 3)
    al = align_colmap_to_gt(exp, pycolmap.Reconstruction(str(exp / "sparse" / "0")))
    print(f"位置合わせ RMSE {al['rmse']*100:.2f}cm  ok={al['ok']}")
    wires, R = wire_polylines(exp)
    fr = free_points(exp, wires, a.n_free, a.free_margin, np.random.default_rng(a.seed))
    print(f"電線 {len(wires)} 本・計 {sum(len(w) for w in wires) * STEP:.1f}m（R={R*1000:.0f}mm）／空き点 殻 {len(fr['shell'])}・箱 {len(fr['box'])}")
    rows = []
    for m in a.models:
        ply = exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply"
        mu, cov, op, smax = load_gaussians(ply, al)
        row = {"exp": exp.name, "model": m, "n_gauss": len(mu)}
        for opk, keep in (("", np.ones(len(mu), bool)), ("_op01", op > 0.1)):
            gaps, cov_len, tot = [], 0, 0
            for w in wires:
                hit = inside_any(w, mu[keep], cov[keep], smax[keep], thr, infl2=R ** 2)
                cov_len += hit.sum()
                tot += len(hit)
                # 覆われていない連続区間の長さ
                run = 0
                for h in np.r_[hit, True]:
                    if h:
                        if run:
                            gaps.append(run * STEP)
                        run = 0
                    else:
                        run += 1
            gaps = np.array(gaps) if gaps else np.zeros(1)
            row[f"wire_cover{opk}"] = cov_len / tot
            row[f"wire_maxgap_m{opk}"] = float(gaps.max())
            row[f"wire_ngap30{opk}"] = int((gaps > GAP_DRONE).sum())
            row[f"wire_gaplen30_m{opk}"] = float(gaps[gaps > GAP_DRONE].sum())
            for name, Q in fr.items():
                row[f"false_occ_{name}{opk}"] = float(inside_any(Q, mu[keep], cov[keep], smax[keep], thr).mean())
                row[f"false_occ_{name}_kappa{opk}"] = float(inside_any(Q, mu[keep], cov[keep], smax[keep], thr, infl2=a.kappa ** 2).mean())
        rows.append(row)
        print(f"{m}: 電線被覆 {row['wire_cover']:.3f}・最大隙間 {row['wire_maxgap_m']:.2f}m・30cm超の隙間 {row['wire_ngap30']} 個"
              f"／誤占有 殻 {row['false_occ_shell']:.4f}・箱 {row['false_occ_box']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else exp / "gt" / "map_eval.csv"
    df.to_csv(out, index=False, mode="a" if a.append and out.exists() else "w", header=not (a.append and out.exists()))
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

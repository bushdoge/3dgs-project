# 「空が写っている画素の視線上は何も無い」ことを使って、正解の3D形状なしで電線・空の高さに「確実に空いている点」を作り、
# 3DGS 地図がそこを塞いでいないかを測る（研究メモ §9aw）。実データ（railside/street）用。合成で手法の確かさを先に確認する。
#   1) 空のマスク：wire_extrap.sky_and_wire の判定（青みが強く明るい）を --erode px 縮め、電線・暗い細線の候補（ブラックハット >--bh-thr）を
#      --wire-dilate px 広げて除き、画像端 --border px も除く（安全側）
#   2) 空の画素（--stride px おき）の視線上に、カメラから --dmin〜--dmax m の点を --dstep m おきに置く → 確実に空いている点
#   3) 測るもの（モデル×不透明度の足切りごと）
#      (a) 空き点のうち地図で障害物扱いの割合。楕円体（1σ、Σ+κ²I）と、Splat-Nav の格子と同じ外接箱（1σ の軸平行外接箱＋κ）の2通り。
#          高さ帯（カメラの平均高さからの高さ）別
#      (b) 空き点を --cell m の格子にして、空き点のセルだけでつながる2点（--min-sep m 以上離れた組）の間を、地図の格子で A*。
#          空き点のセルだけを通る経路が存在するので、A* が失敗すれば地図が確実に空いている空間を塞いでいる
#   合成（--mode synth）では追加で、空き点が本当に空いているかを電線の中心線からの距離で確かめ、空のマスクの正しさを GT 深度（空＝無限遠）で測る。
# 実行は tools/splatnav/venv の python（Splat-Nav の格子のため）。
# 使い方: tools/splatnav/venv/bin/python sky_carve.py --mode real <exp> <model_dir名> [...]
#   → <exp>/sky_carve.csv（実データ）/ <exp>/gt/sky_carve.csv（合成）、計画は *_astar.csv
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import dijkstra3d
import numpy as np
import pandas as pd
import pycolmap
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "real"))
import splatnav_wires as SW              # noqa: E402
from map_eval import inside_any          # noqa: E402
from real_splatnav import camera_frame   # noqa: E402
from wire_extrap import sky_and_wire     # noqa: E402
from scene_stats import read_depth       # noqa: E402

BANDS = [(-99, 2), (2, 5), (5, 10), (10, 99)]   # カメラ平均高さからの高さ[m]


def safe_sky(bgr, a, obs_uv=None):
    """obs_uv：この画像で3D点を持つ SfM 特徴点の画素座標。空には特徴点が出ないので、特徴点を一定密度以上含む空候補の領域は空でないとみなす
    （色と上端への連結だけだと、空の直下の青い壁が空とつながる。合成の最悪視点で適合率 0.54）"""
    wm, sky, _ = sky_and_wire(bgr)
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    thin = ((wm > 0) | (bh > a.bh_thr)).astype(np.uint8)
    thin = cv2.dilate(thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.wire_dilate + 1,) * 2))
    # 色だけだと青っぽい壁も空になる（合成の青い建物で適合率 0.39）。空には SfM の特徴点が出ないので、
    # 特徴点の周り --obs-dilate px を空から除いてから、電線で分断されないよう閉じ（--close px）、画像の上端（上 3% の行）につながる領域だけを空とする。
    # 領域ごと捨てる規則（特徴点密度）は、鉄塔や電線に接する実データの空を丸ごと捨ててしまうので使わない
    if obs_uv is not None and len(obs_uv):
        H, W = sky.shape
        ou = np.clip(obs_uv.astype(int), 0, [W - 1, H - 1])
        om = np.zeros((H, W), np.uint8)
        om[ou[:, 1], ou[:, 0]] = 1
        om = cv2.dilate(om, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.obs_dilate + 1,) * 2))
        sky = sky & (1 - om)
    closed = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close, a.close)))
    n, lab = cv2.connectedComponents(closed)
    top = np.unique(lab[: max(1, int(0.03 * lab.shape[0]))])
    top = top[top > 0]
    sky = sky & np.isin(lab, top).astype(np.uint8)
    s = cv2.erode(sky, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.erode + 1,) * 2)) & (1 - thin)
    b = a.border
    s[:b] = 0
    s[-b:] = 0
    s[:, :b] = 0
    s[:, -b:] = 0
    return s


def observations(rec):
    """画像名 → 3D点を持つ特徴点の画素座標"""
    out = {}
    for im in rec.images.values():
        out[im.name] = np.array([p.xy for p in im.points2D if p.has_point3D()]) if len(im.points2D) else np.zeros((0, 2))
    return out


def cameras_synth(exp, obs):
    """合成：GT 座標（m）の (中心, 視線を作る関数, 画像パス, GT 深度パス, 特徴点)"""
    P = json.load(open(exp / "gt" / "poses.json"))
    f, W, H = P["fx"], P["width"], P["height"]
    out = []
    for key, M in sorted(P["frames"].items()):
        M = np.array(M)
        no = key.split("_")[-1]

        def rays(u, v, M=M):
            d = np.stack([(u - W / 2) / f, -(v - H / 2) / f, -np.ones_like(u)], 1)     # OpenGL：−Z が前
            d = (M[:3, :3] @ d.T).T
            return d / np.linalg.norm(d, axis=1, keepdims=True)
        out.append((M[:3, 3], rays, exp / "input" / f"{key}.png", exp / "gt" / "depth" / f"{no}.exr", obs.get(f"{key}.png")))
    return out


def cameras_real(exp, al_rec, obs):
    al, rec = al_rec
    out = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Rcw = cfw.rotation.matrix()
        C = -Rcw.T @ cfw.translation
        Cm = al["s"] * (al["R"] @ C) + al["t"]

        def rays(u, v, Rcw=Rcw, fx=fx, fy=fy, cx=cx, cy=cy):
            d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], 1)         # OpenCV：+Z が前
            d = (al["R"] @ (Rcw.T @ d.T)).T
            return d / np.linalg.norm(d, axis=1, keepdims=True)
        out.append((Cm, rays, exp / "images" / im.name, None, obs.get(im.name)))
    return out


def carve(cams, a, rng):
    pts, sky_prec = [], []
    ds = np.arange(a.dmin, a.dmax + 1e-9, a.dstep)
    for C, rays, img, dep, obs in cams:
        bgr = cv2.imread(str(img))
        s = safe_sky(bgr, a, obs)
        vv, uu = np.nonzero(s[::a.stride, ::a.stride])
        u, v = uu * a.stride + 0.5, vv * a.stride + 0.5
        if dep is not None and Path(dep).exists():     # 合成：空のマスクの正しさ（GT 深度が無限遠＝空）
            D = read_depth(dep)
            sky_prec.append(float(np.mean(~np.isfinite(D[vv * a.stride, uu * a.stride]) | (D[vv * a.stride, uu * a.stride] > 1e4))))
        d = rays(u, v)
        P = C[None, None, :] + d[:, None, :] * ds[None, :, None]
        pts.append(P.reshape(-1, 3))
    P = np.concatenate(pts)
    if len(P) > a.max_points:
        P = P[rng.choice(len(P), a.max_points, replace=False)]
    return P, (float(np.mean(sky_prec)) if sky_prec else np.nan)


def inside_any_aabb(Q, mu, half, kappa, chunk=20000):
    """Splat-Nav の格子と同じ外接箱（1σ の軸平行外接箱＋κ）に入るか（格子に丸めない版）"""
    hit = np.zeros(len(Q), bool)
    tree = cKDTree(Q)
    r = np.linalg.norm(half, axis=1) + kappa
    lo, hi = Q.min(0), Q.max(0)
    cand = np.nonzero(((mu + half + kappa > lo) & (mu - half - kappa < hi)).all(1))[0]
    for s in range(0, len(cand), chunk):
        g = cand[s:s + chunk]
        nb = tree.query_ball_point(mu[g], r[g], workers=-1)
        for gi, idx in zip(g, nb):
            if not idx:
                continue
            idx = np.asarray(idx)
            ok = np.all(np.abs(Q[idx] - mu[gi]) <= half[gi] + kappa, 1)
            hit[idx[ok]] = True
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth"], required=True)
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--cam-height", type=float, default=1.4)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--erode", type=int, default=4)
    ap.add_argument("--wire-dilate", type=int, default=6)
    ap.add_argument("--bh-thr", type=int, default=12)
    ap.add_argument("--border", type=int, default=8)
    ap.add_argument("--obs-dilate", type=int, default=40, help="SfM 特徴点の周り何 px を空から除くか")
    ap.add_argument("--close", type=int, default=9, help="電線で分断された空をつなぐ閉処理の大きさ[px]")
    ap.add_argument("--dmin", type=float, default=0.5)
    ap.add_argument("--dmax", type=float, default=30.0)
    ap.add_argument("--dstep", type=float, default=0.25)
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--cell", type=float, default=0.2, help="(b) の格子[m]")
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--min-sep", type=float, default=5.0)
    ap.add_argument("--opacity-thr", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    rng = np.random.default_rng(a.seed)
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    if a.mode == "synth":
        al = SW.align_colmap_to_gt(exp, rec)
        cams = cameras_synth(exp, observations(rec))
        out = exp / "gt" / "sky_carve.csv"
    else:
        al, _, _ = camera_frame(rec, a.cam_height)
        cams = cameras_real(exp, (al, rec), observations(rec))
        out = exp / "sky_carve.csv"
    Cs = np.array([c[0] for c in cams])
    z0 = Cs[:, 2].mean()
    t0 = time.time()
    P, sky_prec = carve(cams, a, rng)
    h = P[:, 2] - z0
    keep = P[:, 2] > 0.2 if a.mode == "synth" else np.ones(len(P), bool)    # 合成は地面（z=0）より上だけ
    P, h = P[keep], h[keep]
    print(f"空き点 {len(P)} 個（{time.time()-t0:.0f}s）・高さ帯 " +
          " / ".join(f"{lo}〜{hi}m: {np.sum((h >= lo) & (h < hi))}" for lo, hi in BANDS) +
          (f"・空マスクの適合率 {sky_prec:.4f}" if not np.isnan(sky_prec) else ""), flush=True)
    valid = {}
    if a.mode == "synth":    # 空き点が電線の中に入っていないか（電線の中心線から R 以内＝誤り、R+κ 以内＝機体が触れる）
        wires, R = SW.wire_polylines(exp)
        d = cKDTree(np.concatenate(wires)).query(P)[0]
        valid = {"in_wire": float(np.mean(d < R)), "near_wire": float(np.mean(d < R + a.kappa)), "min_wire_dist_m": float(d.min())}
        print(f"検証：電線の中 {valid['in_wire']:.5f}・電線に κ 以内 {valid['near_wire']:.5f}・最小距離 {valid['min_wire_dist_m']*100:.1f}cm", flush=True)

    # (b) 空き点の格子と、その中だけでつながる組
    lower = P.min(0) - a.cell
    res = tuple(np.ceil((P.max(0) + a.cell - lower) / a.cell).astype(int))
    fi = np.floor((P - lower) / a.cell).astype(int)
    known = np.zeros(res, bool)
    known[fi[:, 0], fi[:, 1], fi[:, 2]] = True
    lab, n_lab = ndimage.label(known)     # 6 近傍
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    comp = np.argwhere(lab == big)
    pairs = []
    while len(pairs) < a.n_pairs:
        i, j = rng.integers(0, len(comp), 2)
        if np.linalg.norm(comp[i] - comp[j]) * a.cell >= a.min_sep:
            pairs.append((comp[i], comp[j]))
    centers = lower + a.cell * (np.argwhere(np.ones(res, bool)) + 0.5)
    print(f"(b) 格子 {res}・既知の空きセル {known.sum()}・最大の連結成分 {len(comp)} セル", flush=True)

    rows, arows = [], []
    for m in a.models:
        mu, q, sc, op = SW.load_ply_gt(exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply", al)
        Rw = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        scc = np.maximum(sc, SW.MIN_SCALE)
        cov = (Rw * scc[:, None, :] ** 2) @ Rw.transpose(0, 2, 1)
        half = np.sqrt(np.einsum("nij,nj->ni", Rw ** 2, scc ** 2))
        for t_op in a.opacity_thr:
            k = op >= t_op
            hit_e = inside_any(P, mu[k], cov[k], scc[k].max(1), 1.0, infl2=a.kappa ** 2, chunk=5000)
            hit_b = inside_any_aabb(P, mu[k], half[k], a.kappa)
            row = {"exp": exp.name, "model": m, "opacity_thr": t_op, "n_points": len(P), "sky_precision": sky_prec, **valid,
                   "occ_ellip": float(hit_e.mean()), "occ_aabb": float(hit_b.mean())}
            for lo, hi in BANDS:
                s = (h >= lo) & (h < hi)
                row[f"occ_ellip_h{lo}_{hi}"] = float(hit_e[s].mean()) if s.any() else np.nan
                row[f"occ_aabb_h{lo}_{hi}"] = float(hit_b[s].mean()) if s.any() else np.nan
            rows.append(row)
            # (b) 地図の格子（楕円体／外接箱）で A*
            for gname in ("ellip", "aabb"):
                if gname == "ellip":
                    occ = inside_any(centers, mu[k], cov[k], scc[k].max(1), 1.0, infl2=a.kappa ** 2, chunk=5000).reshape(res)
                else:
                    occ = inside_any_aabb(centers, mu[k], half[k], a.kappa).reshape(res)
                ok_n = 0
                for pi, (s_, g_) in enumerate(pairs):
                    if occ[tuple(s_)] or occ[tuple(g_)]:
                        path = None
                        ends_blocked = True
                    else:
                        path = dijkstra3d.binary_dijkstra(np.ascontiguousarray(occ), s_, g_, connectivity=6, background_color=1)
                        path = path if len(path) else None
                        ends_blocked = False
                    ok_n += path is not None
                    arows.append({"exp": exp.name, "model": m, "opacity_thr": t_op, "grid": gname, "pair": pi,
                                  "ok": path is not None, "ends_blocked": ends_blocked, "occ_grid": float(occ.mean())})
                row[f"astar_ok_{gname}"] = ok_n / len(pairs)
            print(f"[{m} thr={t_op}] 空き点の占有 楕円体 {row['occ_ellip']:.3f}・外接箱 {row['occ_aabb']:.3f}（高さ帯 " +
                  " / ".join(f"{row[f'occ_ellip_h{lo}_{hi}']:.3f}" for lo, hi in BANDS) +
                  f"）・A* 成功 楕円体 {row['astar_ok_ellip']:.2f} 外接箱 {row['astar_ok_aabb']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(arows).to_csv(str(out).replace(".csv", "_astar.csv"), index=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

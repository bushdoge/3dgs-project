# 実データの電線の3D位置を、多視点の2D検出からボクセル投票で復元する（研究メモ §9ay）。正解の3D形状が無い実データで
# 「電線の見逃し・衝突」を測るための準備。合成で GT の電線と比べて精度を確かめてから使う。
#   1) 各画像：空を背景にした電線の画素（wire_extrap.sky_and_wire。周囲 21px の 85% 以上が空の暗い細線）と、色で判定した空の画素
#   2) 粗い格子（--coarse m）：各ボクセル中心を全視点に投影し、
#        電線の票  = 投影先から最寄りの電線画素までの距離 ≤ ボクセルの見かけ半径＋--tol px
#        反対の票  = 投影先が空の画素で、最寄りの電線画素が（見かけ半径＋--tol＋--sky-margin）px より遠い（視線が空まで抜けている）
#      電線の票 ≥ --min-hits かつ 電線の票 ≥ --ratio ×（電線＋反対）のボクセルを残す
#   3) 残ったボクセルの中を細かい格子（--fine m）で同じ投票 → 電線の3D点群
# 座標：合成は GT 座標（m）、実データは real_splatnav.camera_frame（m、上が +z）。飛行空域の評価スクリプトと同じ座標。
# 出力：<exp>/gt/wire3d.npz（合成）／<exp>/wire3d.npz（実データ）に points（N×3）・hits・ratio。合成では GT との距離で精度・再現率を表示。
# 使い方: python3 wire3d.py --mode synth|real <exp> [--coarse 0.2 --fine 0.04]
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "real"))
from wire_extrap import sky_and_wire     # noqa: E402


def detect_bright(bgr, bh_thr=20, bright=150):
    """明るい背景の上の暗い細線（青さを問わない）。合成の地平線近くの白い空でも拾える。
    ブラックハット（9px）> bh_thr かつ 周囲 21px の明るさの中央値相当（ぼかし）> bright"""
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bg = cv2.dilate(v, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))       # 細線を埋めた背景
    smooth = cv2.blur(np.abs(cv2.Laplacian(cv2.GaussianBlur(bg, (5, 5), 0), cv2.CV_32F)), (21, 21)) < 4
    w = ((bh > bh_thr) & (cv2.blur(bg, (21, 21)) > bright) & smooth).astype(np.uint8)
    return cv2.morphologyEx(w, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def load_views(exp, rec, to_frame, detector="color"):
    """各視点：world（出力座標）→ 画素の射影に必要なものと、電線までの距離画像・空のマスク。
    detector：color＝wire_extrap.sky_and_wire（青い空の上の暗線）／bright＝detect_bright／gtmask＝合成の GT 細線マスク（幾何の検証用）"""
    views = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Rcw, tcw = cfw.rotation.matrix(), np.asarray(cfw.translation)
        # 出力座標 X = s R x + t（x は COLMAP）→ x = R^T (X − t)/s → カメラ座標 = Rcw x + tcw
        s, R, t = to_frame
        A = Rcw @ R.T / s
        b = tcw - A @ t
        p = exp / "images" / im.name
        if not p.exists():
            p = exp / "input" / im.name
        bgr = cv2.imread(str(p))
        wm, sky, _ = sky_and_wire(bgr)
        if detector == "bright":
            wm = detect_bright(bgr)
        elif detector == "gtmask":
            no = im.name.split("_")[-1].split(".")[0]
            wm = (cv2.imread(str(exp / "gt" / "mask_thin" / f"{no}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if detector != "color":     # 空：色の判定に加え、明るく平らな背景も空とみなす（白っぽい空）
            v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            flat = cv2.blur(np.abs(cv2.Laplacian(v, cv2.CV_32F)), (15, 15)) < 3
            sky = (sky.astype(bool) | ((v > 170) & flat)).astype(np.uint8)
        dt = cv2.distanceTransform((1 - wm).astype(np.uint8), cv2.DIST_L2, 5)
        views.append((A, b, fx, fy, cx, cy, dt, sky.astype(bool), wm.sum()))
    return views


def vote(P, half, views, a):
    hits = np.zeros(len(P), np.int32)
    miss = np.zeros(len(P), np.int32)
    for A, b, fx, fy, cx, cy, dt, sky, nw in views:
        if nw == 0:
            continue
        pc = P @ A.T + b
        z = pc[:, 2]
        ok = z > 0.5
        u = np.full(len(P), -1.0)
        v = np.full(len(P), -1.0)
        u[ok] = fx * pc[ok, 0] / z[ok] + cx
        v[ok] = fy * pc[ok, 1] / z[ok] + cy
        H, W = dt.shape
        ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        idx = np.nonzero(ok)[0]
        ui, vi = u[idx].astype(int), v[idx].astype(int)
        r = fx * half / z[idx] + a.tol
        d = dt[vi, ui]
        hits[idx[d <= r]] += 1
        miss[idx[sky[vi, ui] & (d > r + a.sky_margin)]] += 1
    return hits, miss


def grid(lo, hi, step):
    ax = [np.arange(lo[k] + step / 2, hi[k], step) for k in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return np.stack([X, Y, Z], -1).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "real"], required=True)
    ap.add_argument("exp")
    ap.add_argument("--coarse", type=float, default=0.2)
    ap.add_argument("--fine", type=float, default=0.04)
    ap.add_argument("--tol", type=float, default=2.0, help="電線の票の許容[px]（ボクセルの見かけ半径に足す）")
    ap.add_argument("--sky-margin", type=float, default=3.0)
    ap.add_argument("--min-hits", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--tight-tol", type=float, default=1.0, help="芯線抽出の投票の許容[px]")
    ap.add_argument("--nms", type=float, default=0.06, help="芯線抽出の近傍半径[m]")
    ap.add_argument("--nms-frac", type=float, default=0.95)
    ap.add_argument("--views", choices=["all", "even", "odd"], default="all")
    ap.add_argument("--cam-height", type=float, default=1.4)
    ap.add_argument("--detector", choices=["color", "bright", "gtmask"], default="bright")
    ap.add_argument("--hmin", type=float, default=1.0, help="探す高さ：カメラ平均高さ＋hmin〜hmax [m]")
    ap.add_argument("--hmax", type=float, default=12.0)
    ap.add_argument("--xy-margin", type=float, default=12.0)
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    if a.mode == "synth":
        from wbgeom import align_colmap_to_gt
        al = align_colmap_to_gt(exp, rec)
        out = exp / "gt" / f"wire3d_{a.detector}{'' if a.views == 'all' else '_' + a.views}.npz"
    else:
        from real_splatnav import camera_frame
        al, _, _ = camera_frame(rec, a.cam_height)
        out = exp / f"wire3d_{a.detector}{'' if a.views == 'all' else '_' + a.views}.npz"
    to_frame = (al["s"], al["R"], al["t"])
    t0 = time.time()
    views = load_views(exp, rec, to_frame, a.detector)
    if a.views == "even":      # 偶数番で復元し、奇数番で検証する（--views even の後に holdout_check）
        held = views[1::2]
        views = views[::2]
    elif a.views == "odd":     # 奇数番だけで復元（§9bd：偶数番で作った電線で守った後処理を、別の写真の電線で評価する）
        held = views[::2]
        views = views[1::2]
    C = np.array([al["s"] * (al["R"] @ (-(v.cam_from_world() if callable(v.cam_from_world) else v.cam_from_world).rotation.matrix().T
                                         @ np.asarray((v.cam_from_world() if callable(v.cam_from_world) else v.cam_from_world).translation))) + al["t"]
                  for v in sorted(rec.images.values(), key=lambda i: i.name)])
    z0 = C[:, 2].mean()
    lo = np.r_[C[:, :2].min(0) - a.xy_margin, z0 + a.hmin]
    hi = np.r_[C[:, :2].max(0) + a.xy_margin, z0 + a.hmax]
    if a.mode == "synth":
        lo[2] = max(lo[2], 0.3)
    print(f"視点 {len(views)}（電線画素あり {sum(v[-1] > 0 for v in views)}）・探索範囲 {np.round(hi - lo, 1)}m（{time.time()-t0:.0f}s）", flush=True)
    # 粗い投票
    P = grid(lo, hi, a.coarse)
    keep = np.zeros(len(P), bool)
    for s in range(0, len(P), 2_000_000):
        h, m = vote(P[s:s + 2_000_000], a.coarse * 0.87, views, a)
        keep[s:s + 2_000_000] = (h >= a.min_hits) & (h >= a.ratio * (h + m))
    Pc = P[keep]
    print(f"粗い格子 {len(P)} → 残り {len(Pc)}（{time.time()-t0:.0f}s）", flush=True)
    # 細かい投票
    k = int(round(a.coarse / a.fine))
    off = (np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) * a.fine - a.coarse / 2
    pts, hs, rs = [], [], []
    for s in range(0, len(Pc), 20000):
        F = (Pc[s:s + 20000, None, :] + off[None]).reshape(-1, 3)
        h, m = vote(F, a.fine * 0.87, views, a)
        sel = (h >= a.min_hits) & (h >= a.ratio * (h + m))
        pts.append(F[sel])
        hs.append(h[sel])
        rs.append(h[sel] / np.maximum(h[sel] + m[sel], 1))
    W = np.concatenate(pts) if pts else np.zeros((0, 3))
    H = np.concatenate(hs) if hs else np.zeros(0)
    Rr = np.concatenate(rs) if rs else np.zeros(0)
    print(f"細かい格子 → {len(W)} 点（太い管状）（{time.time()-t0:.0f}s）", flush=True)
    # 4) 芯線の抽出：点そのものの大きさ 0・許容 --tight-tol px で投票し直し、半径 --nms m 内で票が最大に近い点だけ残す
    if len(W):
        a2 = argparse.Namespace(tol=a.tight_tol, sky_margin=a.sky_margin)
        h2, m2 = vote(W, 0.0, views, a2)
        sc = h2 - m2
        tree = cKDTree(W)
        nb = tree.query_ball_point(W, a.nms, workers=-1)
        locmax = np.array([sc[i] >= a.nms_frac * sc[j].max() for i, j in enumerate(nb)])
        sel = locmax & (h2 >= a.min_hits)
        W, H, Rr = W[sel], h2[sel], (h2[sel] / np.maximum(h2[sel] + m2[sel], 1))
    np.savez(out, points=W, hits=H, ratio=Rr, scale=al["s"])
    print(f"電線の点 {len(W)}（{time.time()-t0:.0f}s）→ {out}", flush=True)
    if a.views != "all" and len(W):
        # 使っていない奇数番の視点に投影し、写る点のうち検出した電線の画素から 3px 以内に落ちる割合（視点ごとの中央値）
        fr = []
        for A, b, fx, fy, cx, cy, dt, sky, nw in held:
            pc = W @ A.T + b
            ok = pc[:, 2] > 0.5
            u = fx * pc[ok, 0] / pc[ok, 2] + cx
            v = fy * pc[ok, 1] / pc[ok, 2] + cy
            H_, W_ = dt.shape
            ins = (u >= 0) & (u < W_ - 1) & (v >= 0) & (v < H_ - 1)
            if ins.sum() > 50:
                fr.append(float(np.mean(dt[v[ins].astype(int), u[ins].astype(int)] <= 3)))
        print(f"  未使用の視点 {len(fr)} 枚での一致：検出した電線から 3px 以内に落ちる点の割合 中央値 {np.median(fr):.3f}（四分位 {np.percentile(fr, 25):.3f}〜{np.percentile(fr, 75):.3f}）")
    if a.mode == "synth" and len(W):
        from map_eval import wire_polylines
        wires, R = wire_polylines(exp)
        G = np.concatenate(wires)
        d_pred = cKDTree(G).query(W)[0]
        d_gt = cKDTree(W).query(G)[0]
        for tol in (0.05, 0.10, 0.20):
            print(f"  許容 {tol*100:.0f}cm：精度（点が GT 電線の近く）{np.mean(d_pred < tol):.3f}・再現率（GT 電線の長さのうち点がある）{np.mean(d_gt < tol):.3f}")


if __name__ == "__main__":
    main()

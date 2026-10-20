# 実データ（GT の3D形状なし）で、3DGS 地図が「確実に空いている空間」を塞ぐかを Splat-Nav（Splat-Plan 公開実装）で測る（研究メモ §9av）。
# 撮影したカメラの位置は、撮影者がそこを通ったので確実に空いている。そこで
#   (a) カメラ軌道（連続フレームの中心を結んだ折れ線）のうち、1σ楕円体（機体半径 κ で膨らませる）に入っている割合（＝空いているはずの所が塞がれている）
#   (b) カメラ位置 i → j を Splat-Plan で計画したときの成功率（歩いた軌道があるので、正しい地図なら必ず経路がある）と経路長の歩行距離比
#   (c) 軌道を塞いでいるガウシアンの性質（最大軸の長さ・不透明度・最寄りカメラからの距離）
# を出す。不透明度の足切りごと。
# 尺度：COLMAP 座標には尺度が無いので、カメラの地面からの高さ（各カメラ真下 1 単位以内の SfM 点の 5 パーセンタイル）を
#       --cam-height [m]（手持ち撮影の既定 1.4m）とみなして m に直す。上方向はカメラの −y 軸の中央値。
# 実行は tools/splatnav/venv の python（splatnav_wires.py と同じ）。
# 使い方: tools/splatnav/venv/bin/python real/real_splatnav.py <exp> <model_dir名> [...] [--kappa 0.1] [--opacity-thr 0 0.1 0.5]
#   → <exp>/real_splatnav.csv（計画の行）と <exp>/real_splatnav_path.csv（モデル×閾値ごとの軌道の塞がれ方）
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
import torch
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import splatnav_wires as SW                        # noqa: E402  （Splat-Nav の読み込み・スタブ・Map・plan）
from map_eval import inside_any                    # noqa: E402

STEP = 0.02     # 軌道のサンプル間隔[m]


def camera_frame(rec, cam_height):
    """COLMAP → m・上方向を +z にそろえた座標への変換と、カメラ中心（名前順）"""
    ims = sorted(rec.images.values(), key=lambda i: i.name)
    cfw = [i.cam_from_world() if callable(i.cam_from_world) else i.cam_from_world for i in ims]
    Rm = np.array([c.rotation.matrix() for c in cfw])
    tv = np.array([c.translation for c in cfw])
    C = -np.einsum("nji,nj->ni", Rm, tv)
    up = -np.median(Rm[:, 1, :], 0)
    up /= np.linalg.norm(up)
    P = np.array([p.xyz for p in rec.points3D.values()])
    hs = []
    for c in C:
        d = P - c
        h = d @ up
        horiz = np.linalg.norm(d - np.outer(h, up), axis=1)
        sel = (horiz < 1.0) & (h < 0)
        if sel.sum() > 20:
            hs.append(-np.percentile(h[sel], 5))
    h_units = float(np.median(hs))
    s = cam_height / h_units
    # 進行方向（カメラ軌道の主成分を水平面に射影）を x、上を z
    d0 = C[-1] - C[0]
    x = d0 - (d0 @ up) * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    R = np.stack([x, y, up])           # 行が新しい軸
    t = -s * (R @ C.mean(0))
    return {"s": s, "R": R, "t": t, "h_units": h_units, "n_ground": len(hs)}, (s * (R @ C.T)).T + t, [i.name for i in ims]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--cam-height", type=float, default=1.4)
    ap.add_argument("--kappa", type=float, default=0.1, help="機体半径[m]")
    ap.add_argument("--cell", type=float, default=0.1, help="A* の格子間隔[m]")
    ap.add_argument("--margin", type=float, default=1.0, help="飛行空域＝カメラ軌道の外接箱＋この余白[m]")
    ap.add_argument("--opacity-thr", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    SW.use_fast_voxel()     # 元実装のボクセル化は大きな空のガウシアン×細かい格子で爆発する（合成で一致を確認済み）
    exp = Path(a.exp).resolve()
    al, C, names = camera_frame(pycolmap.Reconstruction(str(exp / "sparse" / "0")), a.cam_height)
    print(f"尺度 {al['s']:.3f} m/単位（カメラ高 {al['h_units']:.3f} 単位＝{a.cam_height}m、地面点のあるカメラ {al['n_ground']}）"
          f"・軌道長 {np.linalg.norm(np.diff(C, axis=0), axis=1).sum():.1f}m")
    lower, upper = C.min(0) - a.margin, C.max(0) + a.margin
    SW.LOWER, SW.UPPER = lower, upper
    res = torch.tensor(np.ceil((upper - lower) / a.cell).astype(int))
    voxel = {"lower_bound": torch.tensor(lower, dtype=torch.float32), "upper_bound": torch.tensor(upper, dtype=torch.float32),
             "resolution": res}
    robot = {"radius": a.kappa, "vmax": 0.1, "amax": 0.1}
    sp = SW.SplinePlanner(spline_deg=6, device=SW.DEV)
    # 軌道の密なサンプルと、組（i<j、10〜40 フレーム離れた組）
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    sq = np.arange(0, cum[-1], STEP)
    path = np.stack([np.interp(sq, cum, C[:, k]) for k in range(3)], 1)
    rng = np.random.default_rng(a.seed)
    I = rng.integers(0, len(C) - 10, a.n_pairs)
    J = np.minimum(I + rng.integers(10, 41, a.n_pairs), len(C) - 1)
    thr = chi2.ppf(0.2, 3)       # Splat-Nav は scale そのもの（≈1σ）。χ²₃(0.2)≈1.005 で同じ
    rows, prow = [], []
    for m in a.models:
        mu, q, sc, op = SW.load_ply_gt(exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply", al)
        c = SW.crop(mu, sc, pad=1.0)
        from scipy.spatial.transform import Rotation
        Rw = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        for t_op in a.opacity_thr:
            k = c & (op >= t_op)
            cov = (Rw[k] * np.maximum(sc[k], SW.MIN_SCALE)[:, None, :] ** 2) @ Rw[k].transpose(0, 2, 1)
            smax = sc[k].max(1)
            hit = inside_any(path, mu[k], cov, smax, thr, infl2=a.kappa ** 2)
            # 軌道を塞いでいるガウシアンの性質：塞がれた軌道点ごとに最も近い（マハラノビス最小）中心を探す代わりに、
            # 軌道点から (1σ最大半径+κ) 以内の中心を集めて要約する
            from scipy.spatial import cKDTree
            tr = cKDTree(path[hit]) if hit.any() else None
            blk = np.zeros(k.sum(), bool)
            if tr is not None:
                dd = tr.query(mu[k])[0]
                blk = dd < smax + a.kappa
            dcam = cKDTree(C).query(mu[k][blk])[0] if blk.any() else np.array([np.nan])
            mp = SW.Map(mu[k], q[k], sc[k])
            t0 = time.time()
            planner = SW.SplatPlan(mp, robot, voxel, sp, SW.DEV)
            grid = planner.gsplat_voxel
            idx = grid.get_indices
            occ_path = float(np.mean([bool(grid.non_navigable_grid[tuple(idx(torch.tensor(p, dtype=torch.float32)).tolist())])
                                      for p in path[::10]]))
            prow.append({"exp": exp.name, "model": m, "opacity_thr": t_op, "n_gauss": int(k.sum()), "scale_m_per_unit": al["s"],
                         "path_blocked": float(hit.mean()), "path_blocked_voxel": occ_path,
                         "occ_voxel": float(grid.non_navigable_grid.float().mean()),
                         "blk_n": int(blk.sum()), "blk_size_med_m": float(np.median(smax[blk])) if blk.any() else np.nan,
                         "blk_op_med": float(np.median(op[k][blk])) if blk.any() else np.nan,
                         "blk_dcam_med_m": float(np.nanmedian(dcam))})
            print(f"[{m} thr={t_op}] 軌道の塞がれ {hit.mean():.3f}（ボクセル {occ_path:.3f}）・占有 {prow[-1]['occ_voxel']:.3f}・"
                  f"塞ぐガウシアン {blk.sum()} 個（最大軸中央 {prow[-1]['blk_size_med_m']:.2f}m・不透明度 {prow[-1]['blk_op_med']:.2f}・"
                  f"カメラから {prow[-1]['blk_dcam_med_m']:.2f}m）（{time.time()-t0:.0f}s）", flush=True)
            for p_i, (i, j) in enumerate(zip(I, J)):
                traj, status, dt = SW.plan(planner, C[i], C[j])
                row = {"exp": exp.name, "model": m, "opacity_thr": t_op, "pair": p_i, "i": int(i), "j": int(j),
                       "walk_m": float(cum[j] - cum[i]), "status": status, "plan_s": round(dt, 2)}
                if traj is not None:
                    row["length"] = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
                    row["len_ratio_walk"] = row["length"] / row["walk_m"]
                rows.append(row)
            d = pd.DataFrame([r for r in rows if r["model"] == m and r["opacity_thr"] == t_op])
            print(f"   計画の成功 {(d.status == 'ok').sum()}/{len(d)}", flush=True)
    pd.DataFrame(rows).to_csv(exp / "real_splatnav.csv", index=False)
    pd.DataFrame(prow).to_csv(exp / "real_splatnav_path.csv", index=False)
    print(f"保存: {exp / 'real_splatnav.csv'}, {exp / 'real_splatnav_path.csv'}")


if __name__ == "__main__":
    main()

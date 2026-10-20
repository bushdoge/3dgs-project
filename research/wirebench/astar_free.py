# Splat-Nav（Splat-Plan）の第1段＝ボクセル格子上の A* だけを取り出し、3DGS 地図が「通れるはずの所」を塞ぐかを測る（研究メモ §9av）。
# Splat-Plan は A* で粗い経路を作れないと計画全体が失敗する（必要条件）。第2段（凸多面体の通路と Bezier の QP）は実データでは
# 1組 10 分超かかるので省く。格子は2通りで作り、「地図が悪い」と「Splat-Nav の格子化が保守的すぎる」を切り分ける。
#   aabb  : Splat-Nav と同じ。各ガウシアンの1σ楕円体の軸平行外接箱が重なるセルを占有 → 機体半径 κ で膨張（splatnav_wires.fast_navigable_grid。元実装と一致確認済み）
#   ellip : セル中心が楕円体（Σ+κ²I、1σ）の中にあるセルだけ占有（外接箱で膨らませない）
# 課題：
#   --mode real  : 実データ。撮影カメラの位置 i→j（10〜40 フレーム離れた組）。撮影者が歩いた軌道があるので正しい地図なら必ず通れる。
#                  尺度はカメラの地面からの高さ（real/real_splatnav.py の camera_frame、既定 1.4m）
#   --mode synth : 合成。splatnav_wires.py と同じ電線をまたぐ組。A* の経路が GT の電線に κ 以内まで近づいたら衝突
# 実行は tools/splatnav/venv の python。
# 使い方: tools/splatnav/venv/bin/python astar_free.py --mode real <exp> <model_dir名> [...] [--kappa 0.1 --cell 0.1]
#   → <exp>/gt/astar_free.csv（合成）または <exp>/astar_free.csv（実データ）。1行＝モデル×不透明度閾値×格子×組
import argparse
import sys
import time
from pathlib import Path

import dijkstra3d
import numpy as np
import pandas as pd
import pycolmap
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "real"))
import splatnav_wires as SW              # noqa: E402
from map_eval import inside_any          # noqa: E402
from real_splatnav import camera_frame   # noqa: E402


def grid_centers(lower, cell, res):
    ax = [lower[k] + cell * (np.arange(res[k]) + 0.5) for k in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return np.stack([X, Y, Z], -1).reshape(-1, 3)


def grid_aabb(mp, lower, upper, res, kappa):
    from initialization.grid_utils import GSplatVoxel
    SW.use_fast_voxel()
    v = GSplatVoxel(mp, lower_bound=torch.tensor(lower, dtype=torch.float32), upper_bound=torch.tensor(upper, dtype=torch.float32),
                    resolution=torch.tensor(res), radius=kappa, device=SW.DEV)
    return v.non_navigable_grid.numpy().copy()


def grid_ellip(mu, cov, smax, centers, res, kappa):
    occ = inside_any(centers, mu, cov, smax, 1.0, infl2=kappa ** 2, chunk=5000)
    return occ.reshape(res)


def to_idx(p, lower, cell, res):
    return np.clip(np.floor((p - lower) / cell).astype(int), 0, np.array(res) - 1)


def astar(occ, s, g, centers_free_tree, free_idx, res):
    """Splat-Nav の create_path と同じく、始点・終点が占有なら最寄りの空きセルへ寄せてから 6 近傍の A*（dijkstra3d）"""
    moved = 0
    pts = []
    if len(free_idx) == 0:      # 空域が全部占有（street の素3DGS で起きた）
        return None, 2
    for p in (s, g):
        if occ[tuple(p)]:
            moved += 1
            p = free_idx[centers_free_tree.query(p)[1]]
        pts.append(p)
    path = dijkstra3d.binary_dijkstra(np.ascontiguousarray(occ.astype(bool)), pts[0], pts[1], connectivity=6, background_color=1)
    return (path if len(path) > 0 else None), moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth"], required=True)
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--kappa", type=float, default=None, help="機体半径[m]（既定 real 0.1 / synth 0.05）")
    ap.add_argument("--cell", type=float, default=None, help="格子[m]（既定 real 0.1 / synth 0.05）")
    ap.add_argument("--cam-height", type=float, default=1.4)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--opacity-thr", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    kappa = a.kappa if a.kappa is not None else (0.1 if a.mode == "real" else 0.05)
    cell = a.cell if a.cell is not None else (0.1 if a.mode == "real" else 0.05)
    exp = Path(a.exp).resolve()
    rng = np.random.default_rng(a.seed)
    if a.mode == "real":
        al, C, _ = camera_frame(pycolmap.Reconstruction(str(exp / "sparse" / "0")), a.cam_height)
        lower, upper = C.min(0) - a.margin, C.max(0) + a.margin
        I = rng.integers(0, len(C) - 10, a.n_pairs)
        J = np.minimum(I + rng.integers(10, 41, a.n_pairs), len(C) - 1)
        starts, goals = C[I], C[J]
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        ref = cum[J] - cum[I]                    # 歩いた距離
        wtree, R = None, 0.0
        cam_tree_pts = C
        out = exp / "astar_free.csv"
        print(f"尺度 {al['s']:.3f} m/単位・軌道長 {cum[-1]:.1f}m")
    else:
        al = SW.align_colmap_to_gt(exp, pycolmap.Reconstruction(str(exp / "sparse" / "0")))
        lower, upper = SW.LOWER, SW.UPPER
        starts, goals = SW.make_pairs(a.n_pairs, rng)
        wires, R = SW.wire_polylines(exp)
        wtree = cKDTree(np.concatenate(wires))
        ref = None
        cam_tree_pts = None
        out = exp / "gt" / "astar_free.csv"
    SW.LOWER, SW.UPPER = lower, upper
    res = tuple(np.ceil((upper - lower) / cell).astype(int))
    centers = grid_centers(lower, cell, res)
    print(f"空域 {np.round(upper - lower, 1)}m・格子 {res}（{np.prod(res)/1e6:.1f}M セル）・κ={kappa}m・セル {cell}m", flush=True)

    maps = []
    if a.mode == "synth":   # GT 地図（電線＝半径 R の球の列。単位四元数は Splat-Nav で NaN になるので小さく回す）
        W = np.concatenate(wires)
        maps.append(("gt_wires", -1.0, W, np.tile([np.cos(0.05), np.sin(0.05), 0, 0], (len(W), 1)), np.full((len(W), 3), R)))
    for m in a.models:
        mu, q, sc, op = SW.load_ply_gt(exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply", al)
        c = SW.crop(mu, sc, pad=1.0)
        for t_op in a.opacity_thr:
            k = c & (op >= t_op)
            maps.append((m, t_op, mu[k], q[k], sc[k]))
    rows = []
    gt_len = {}
    for name, t_op, mu, q, sc in maps:
        t0 = time.time()
        mp = SW.Map(mu, q, sc)
        Rw = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        scc = np.maximum(sc, SW.MIN_SCALE)
        cov = (Rw * scc[:, None, :] ** 2) @ Rw.transpose(0, 2, 1)
        grids = {"aabb": grid_aabb(mp, lower, upper, res, kappa), "ellip": grid_ellip(mu, cov, scc.max(1), centers, res, kappa)}
        for gname, occ in grids.items():
            free_idx = np.argwhere(~occ)
            ftree = cKDTree(free_idx) if len(free_idx) else None
            occ_frac = float(occ.mean())
            cam_occ = float(np.mean([occ[tuple(to_idx(p, lower, cell, res))] for p in cam_tree_pts])) if cam_tree_pts is not None else np.nan
            ok_n = 0
            for i, (s, g) in enumerate(zip(starts, goals)):
                si, gi = to_idx(s, lower, cell, res), to_idx(g, lower, cell, res)
                path, moved = astar(occ, si, gi, ftree, free_idx, res)
                row = {"exp": exp.name, "model": name, "opacity_thr": t_op, "grid": gname, "pair": i, "occ": occ_frac,
                       "cam_occ": cam_occ, "endpoint_moved": moved, "ok": path is not None}
                if path is not None:
                    ok_n += 1
                    P = centers.reshape(*res, 3)[path[:, 0], path[:, 1], path[:, 2]]
                    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
                    row["length"] = L
                    if ref is not None:
                        row["len_ratio"] = L / max(ref[i], 1e-6)
                    if wtree is not None:
                        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
                        dense = np.concatenate([np.linspace(p0, p1, max(2, int(np.ceil(d / 0.01)) + 1)) for p0, p1, d in zip(P[:-1], P[1:], seg)])
                        dmin = float(wtree.query(dense)[0].min())
                        row["wire_clear_m"] = dmin - R
                        row["collide"] = bool(dmin < R + kappa)
                        if name == "gt_wires":
                            gt_len[(gname, i)] = L
                        if (gname, i) in gt_len:
                            row["len_ratio"] = L / gt_len[(gname, i)]
                rows.append(row)
            msg = f"[{name} thr={t_op} {gname}] 占有 {occ_frac:.3f}"
            if cam_tree_pts is not None:
                msg += f"・カメラ位置の占有 {cam_occ:.3f}"
            print(msg + f"・A* 成功 {ok_n}/{len(starts)}（{time.time()-t0:.0f}s）", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

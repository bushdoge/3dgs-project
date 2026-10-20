# 3DGS 地図での経路計画に電線がどう効くかを、Splat-Nav の公開実装（Splat-Plan, github.com/chengine/splatnav e996e22、
# tools/splatnav に clone）でそのまま確かめる（研究メモ §9au）。map_eval.py（§9at）の「被覆・誤占有」を、計画の成否で測り直すもの。
#   - 地図：学習済み ply を GT 座標（m）に位置合わせして Splat-Plan に渡す（Mip 系は 3D フィルタ込みの実効スケール。
#     Splat-Plan は各ガウシアンの1σ楕円体を障害物にし、不透明度は読み込むが使わない）。--opacity-thr で不透明度の足切りを振る
#   - 課題：電線（3本）をまたいで反対側へ行く始点・終点の組（電線の手前 y=0.0 → 奥 y=−3.0、高さ 2.6〜4.4m）
#   - 基準：GT の電線だけを半径 R の球の列にした地図で同じ組を計画（正しい地図での経路長）
#   - 判定：計画が失敗したか（A* が通れない／QP が解けない）、得られた軌道が GT の電線に機体半径 κ 以内まで近づいたか（＝衝突）、
#     経路長の GT 地図比（無い壁による遠回り）
# 実行は tools/splatnav/venv の python（torch・open3d・clarabel 等。nerfstudio は使わないのでスタブで置き換える）。GPU 不要（CPU）。
# 使い方: tools/splatnav/venv/bin/python splatnav_wires.py <exp> <model_dir名> [...] [--opacity-thr 0 0.1] [--n-pairs 30]
#   → <exp>/gt/splatnav_wires.csv（1行＝モデル×不透明度閾値×始点終点の組）
import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SPLATNAV = Path("/workspace/tools/splatnav")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPLATNAV))
# Splat-Nav は ply を nerfstudio 経由で読むが、ここでは自前で読むので nerfstudio の import をスタブにする
_ns = types.ModuleType("ns_utils.nerfstudio_utils")
_ns.GaussianSplat = None
_ns.SH2RGB = lambda x: x
sys.modules["ns_utils.nerfstudio_utils"] = _ns
torch.cuda.synchronize = lambda *a, **k: None     # CPU で動かすため

from ellipsoids.covariance_utils import compute_cov       # noqa: E402
from splatplan.splatplan import SplatPlan                  # noqa: E402
from splatplan.spline_utils import SplinePlanner           # noqa: E402
from wbgeom import align_colmap_to_gt                      # noqa: E402
from map_eval import wire_polylines                        # noqa: E402

DEV = torch.device("cpu")
LOWER = np.array([-3.0, -3.3, 1.8])
UPPER = np.array([3.0, 0.3, 5.0])
MIN_SCALE = 1e-4     # 2DGS（厚み0）や極端に薄いガウシアンの下限[m]（Splat-Plan は 1/scale を使う）


class Map:
    """Splat-Plan が使う属性だけを持つ地図（GSplatLoader の代わり）"""
    def __init__(self, means, quats_wxyz, scales):
        self.means = torch.tensor(means, dtype=torch.float32, device=DEV)
        self.rots = torch.tensor(quats_wxyz, dtype=torch.float32, device=DEV)
        self.scales = torch.tensor(np.maximum(scales, MIN_SCALE), dtype=torch.float32, device=DEV)
        self.covs_inv = compute_cov(self.rots, 1 / self.scales)
        self.covs = compute_cov(self.rots, self.scales)
        self.colors = 0.5 * torch.ones_like(self.means)
        self.opacities = torch.ones(len(self.means), device=DEV)


def load_ply_gt(ply, al):
    """ply → GT 座標の中心・回転（wxyz）・スケール[m]・実効不透明度"""
    v = PlyData.read(str(ply))["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3) if f"scale_{i}" in names], 1).astype(np.float64))
    if sc.shape[1] == 2:
        sc = np.concatenate([sc, np.zeros((len(sc), 1))], 1)
    if "filter_3D" in names:
        f2 = np.asarray(v["filter_3D"], np.float64)[:, None] ** 2
        op = op * np.sqrt(np.prod(sc ** 2 / (sc ** 2 + f2), axis=1))
        sc = np.sqrt(sc ** 2 + f2)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)   # wxyz
    Rl = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    Rg = al["R"][None] @ Rl
    qg = Rotation.from_matrix(Rg).as_quat()[:, [3, 0, 1, 2]]
    mu = (al["s"] * (al["R"] @ xyz.T)).T + al["t"]
    return mu, qg, sc * al["s"], op


def crop(mu, sc, pad=1.0):
    r = sc.max(1)
    return np.all((mu + r[:, None] > LOWER - pad) & (mu - r[:, None] < UPPER + pad), axis=1)


def make_pairs(n, rng):
    xs = rng.uniform(-2.3, 2.3, n)
    s = np.c_[xs, np.full(n, 0.0), rng.uniform(2.6, 4.4, n)]
    g = np.c_[np.clip(xs + rng.uniform(-1, 1, n), -2.5, 2.5), np.full(n, -3.0), rng.uniform(2.6, 4.4, n)]
    return s, g


def plan(planner, s, g):
    t0 = time.time()
    try:
        out = planner.generate_path(torch.tensor(s, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        traj = np.asarray(out["traj"])[:, :3]
        ok = bool(out["feasible"])
    except BaseException as e:   # A* 失敗（path=None）や QP 不能（splatplan は素の raise で落ちる）
        return None, f"fail:{type(e).__name__}", time.time() - t0
    return traj, ("ok" if ok else "infeasible"), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("models", nargs="*")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--opacity-thr", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--kappa", type=float, default=0.05, help="機体半径[m]")
    ap.add_argument("--cell", type=float, default=0.05, help="A* の格子間隔[m]")
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--gt-only", action="store_true", help="既存 csv の GT 地図の行だけ計算し直す")
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    al = align_colmap_to_gt(exp, pycolmap.Reconstruction(str(exp / "sparse" / "0")))
    wires, R = wire_polylines(exp)
    W = np.concatenate(wires)
    wtree = cKDTree(W)
    starts, goals = make_pairs(a.n_pairs, np.random.default_rng(a.seed))
    res = torch.tensor(np.round((UPPER - LOWER) / a.cell).astype(int), device=DEV)
    voxel = {"lower_bound": torch.tensor(LOWER, dtype=torch.float32), "upper_bound": torch.tensor(UPPER, dtype=torch.float32),
             "resolution": res}
    robot = {"radius": a.kappa, "vmax": 0.1, "amax": 0.1}     # vmax/amax は公開スクリプトの値
    sp = SplinePlanner(spline_deg=6, device=DEV)

    # 単位四元数ちょうどは Splat-Nav の quaternion_to_angle_axis で NaN になり障害物が消える（等方なので小さく回しても形は同じ）
    qg = np.tile([np.cos(0.05), np.sin(0.05), 0, 0], (len(W), 1))
    maps = [("gt_wires", None, Map(W, qg, np.full((len(W), 3), R)))]
    out = Path(a.out) if a.out else exp / "gt" / "splatnav_wires.csv"
    prev = None
    if a.gt_only:   # 既存の結果の GT 行だけ作り直し、経路長比を付け直す
        prev = pd.read_csv(out)
        prev = prev[prev.model != "gt_wires"]
        a.models = []
    for m in a.models:
        mu, q, sc, op = load_ply_gt(exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply", al)
        c = crop(mu, sc)
        for thr in a.opacity_thr:
            k = c & (op >= thr)
            maps.append((m, thr, Map(mu[k], q[k], sc[k])))
    rows = []
    ref_len = {}
    for name, thr, mp in maps:
        t0 = time.time()
        planner = SplatPlan(mp, robot, voxel, sp, DEV)
        occ = float(planner.gsplat_voxel.non_navigable_grid.float().mean())
        print(f"[{name} thr={thr}] ガウシアン {len(mp.means)}・占有ボクセル {occ:.3f}（{time.time()-t0:.0f}s）", flush=True)
        for i, (s, g) in enumerate(zip(starts, goals)):
            traj, status, dt = plan(planner, s, g)
            row = {"exp": exp.name, "model": name, "opacity_thr": thr, "pair": i, "status": status, "plan_s": round(dt, 2),
                   "occ_voxel": occ, "n_gauss": len(mp.means)}
            if traj is not None:
                L = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
                # 軌道を 1cm 以下に細かくして GT の電線との最短距離（中心線から）を測る
                seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                dense = np.concatenate([np.linspace(p, q_, max(2, int(np.ceil(d / 0.01)) + 1))
                                        for p, q_, d in zip(traj[:-1], traj[1:], seg)])
                dmin = float(wtree.query(dense)[0].min())
                row.update(length=L, wire_clear_m=dmin - R, collide=bool(dmin < R + a.kappa))
                if name == "gt_wires":
                    ref_len[i] = L
                if i in ref_len:
                    row["len_ratio"] = L / ref_len[i]
            rows.append(row)
        d = pd.DataFrame([r for r in rows if r["model"] == name and r["opacity_thr"] == thr])
        okd = d[d.status == "ok"]
        print(f"   成功 {len(okd)}/{len(d)}・衝突 {int(okd.collide.sum()) if len(okd) else 0}・"
              f"経路長比 中央値 {okd.len_ratio.median() if 'len_ratio' in okd and len(okd) else float('nan'):.3f}", flush=True)
    df = pd.DataFrame(rows)
    if prev is not None:
        prev["len_ratio"] = prev.length / prev.pair.map(ref_len)
        df = pd.concat([df, prev], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

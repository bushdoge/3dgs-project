# 実データで Splat-Nav（Splat-Plan の全工程：A* → 凸多面体の通路 → Bezier の QP）を実際に動かし、
# 「確実に空いている空間」の中の2点の間に経路が引けるか・電線に当たらないかを測る（研究メモ §9az）。
#   --prepare : sky_carve.py と同じ方法で空の視線から既知の空き点を作り（全視点）、0.2m 格子の最大連結成分から
#               5m 以上離れた組を作って <exp>/splatnav_full_pairs.npz に保存（正しい地図なら必ず経路がある組）
#   --run     : 1 地図 × 1 組を計画し、結果を 1 行の JSON で <exp>/splatnav_full_g015.jsonl に追記（A* の格子 0.15m）。時間切れは外側の timeout で扱う
# 地図は Splat-Nav 元実装と同じボクセル化（splatnav_wires.fast_navigable_grid。元実装と占有が一致することを確認済み）。
# 機体半径 κ=0.1m、格子 0.2m、vmax=amax=0.1（公開スクリプトの値）、不透明度の足切りなし（論文どおり）。
# 電線の衝突：wire3d.py で復元した電線の点（wire3d_bright.npz）に、軌道の中心が κ+2cm 以内まで近づいたら衝突とみなす。
# 実行は tools/splatnav/venv の python。
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import splatnav_wires as SW          # noqa: E402
import sky_carve as SC               # noqa: E402
from real_splatnav import camera_frame   # noqa: E402
import pycolmap                      # noqa: E402

CELL = 0.2       # 既知の空き点の格子（組を作る）
GRID = 0.15      # Splat-Nav の A* の格子。0.2 だと機体半径 0.1 の膨張核が浮動小数の誤差で 1 か 3 に揺れる（§9az）ので 0.15（必ず 3×3×3）
KAPPA = 0.1


def prepare(exp, n_pairs, seed):
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    al, _, _ = camera_frame(rec, 1.4)
    cams = SC.cameras_real(exp, (al, rec), SC.observations(rec))
    a = argparse.Namespace(stride=12, erode=4, wire_dilate=6, bh_thr=12, border=8, obs_dilate=40, close=9,
                           dmin=0.5, dmax=15.0, dstep=0.25, max_points=400000)
    rng = np.random.default_rng(seed)
    P, _ = SC.carve(cams, a, rng)
    lower = P.min(0) - CELL
    res = np.ceil((P.max(0) + CELL - lower) / CELL).astype(int)
    fi = np.floor((P - lower) / CELL).astype(int)
    known = np.zeros(res, bool)
    known[fi[:, 0], fi[:, 1], fi[:, 2]] = True
    lab, _ = ndimage.label(known)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    comp = np.argwhere(lab == big)
    pairs = []
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, len(comp), 2)
        if np.linalg.norm(comp[i] - comp[j]) * CELL >= 5.0:
            pairs.append((comp[i], comp[j]))
    pts = np.array([[lower + CELL * (s + 0.5), lower + CELL * (g + 0.5)] for s, g in pairs])
    np.savez(exp / "splatnav_full_pairs.npz", lower=lower, upper=lower + CELL * res, pairs=pts, scale=al["s"])
    print(f"{exp.name}: 空き点 {len(P)}・空域 {np.round(CELL * res, 1)}m・組 {len(pts)} → {exp / 'splatnav_full_pairs.npz'}")


def run(exp, model, pair_ids):
    SW.use_fast_voxel()
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    al, _, _ = camera_frame(rec, 1.4)
    z = np.load(exp / "splatnav_full_pairs.npz")
    lower, upper, pairs = z["lower"], z["upper"], z["pairs"]
    SW.LOWER, SW.UPPER = lower, upper
    mu, q, sc, op = SW.load_ply_gt(exp / model / "point_cloud" / "iteration_30000" / "point_cloud.ply", al)
    c = SW.crop(mu, sc, pad=1.0)
    mp = SW.Map(mu[c], q[c], sc[c])
    res = torch.tensor(np.ceil((upper - lower) / GRID).astype(int))
    voxel = {"lower_bound": torch.tensor(lower, dtype=torch.float32), "upper_bound": torch.tensor(upper, dtype=torch.float32),
             "resolution": res}
    t0 = time.time()
    planner = SW.SplatPlan(mp, {"radius": KAPPA, "vmax": 0.1, "amax": 0.1}, voxel, SW.SplinePlanner(spline_deg=6, device=SW.DEV), SW.DEV)
    t_build = time.time() - t0
    wp = exp / "wire3d_bright.npz"
    wtree = cKDTree(np.load(wp)["points"]) if wp.exists() else None
    for pi in pair_ids:
        s, g = pairs[pi]
        traj, status, dt = SW.plan(planner, s, g)
        row = {"exp": exp.name, "model": model, "pair": int(pi), "status": status, "plan_s": round(dt, 1),
               "build_s": round(t_build, 1), "n_gauss": int(c.sum()), "straight_m": float(np.linalg.norm(g - s))}
        if traj is not None:
            row["length"] = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
            if wtree is not None:
                seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                dense = np.concatenate([np.linspace(p0, p1, max(2, int(np.ceil(d / 0.01)) + 1)) for p0, p1, d in zip(traj[:-1], traj[1:], seg)])
                dmin = float(wtree.query(dense)[0].min())
                row["wire_dist_m"] = dmin
                row["wire_collide"] = bool(dmin < KAPPA + 0.02)
        with open(exp / "splatnav_full_g015.jsonl", "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model")
    ap.add_argument("--pairs", type=int, nargs="*")
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    if a.prepare:
        prepare(exp, a.n_pairs, a.seed)
    else:
        run(exp, a.model, a.pairs)


if __name__ == "__main__":
    main()

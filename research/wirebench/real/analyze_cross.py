# 電線をまたぐ組の Splat-Nav の経路（real_splatnav_full.py が保存した splatnav_traj/*.npy）が、電線の所をどう通ったかを分類する（研究メモ §9bb）。
#   経路を 1cm 間隔に細かくし、各点の真上・真下（水平距離 --rh m 以内）に電線の点（wire3d.py の復元）があるかを調べる。
#   ある点では、その近くの電線の点の高さと比べて
#     上：一番上の電線より --dz m 以上高い ／ 下：一番下の電線より --dz m 以上低い ／ 電線の高さの範囲：それ以外（電線の間か、すぐ横）
#   経路全体で一度も電線の真上・真下を通らなければ「電線の所を通らない」（端を回った、または離れた所を通った）。
#   あわせて、経路と電線の最小距離（3次元）と、地面（カメラ平均の高さ −1.4m）からの高さを出す。
# 使い方: python3 real/analyze_cross.py <exp> [--jsonl splatnav_cross_traj.jsonl]
#   → <exp>/splatnav_cross_traj_analysis.csv と要約の表示
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_splatnav import camera_frame   # noqa: E402


def densify(T, step=0.01):
    seg = np.linalg.norm(np.diff(T, axis=0), axis=1)
    return np.concatenate([np.linspace(a, b, max(2, int(np.ceil(d / step)) + 1)) for a, b, d in zip(T[:-1], T[1:], seg)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--jsonl", default="splatnav_cross_traj.jsonl")
    ap.add_argument("--rh", type=float, default=0.3)
    ap.add_argument("--dz", type=float, default=0.1)
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    _, C, _ = camera_frame(pycolmap.Reconstruction(str(exp / "sparse" / "0")), 1.4)
    ground = C[:, 2].mean() - 1.4
    W = np.load(exp / "wire3d_bright.npz")["points"]
    wt3 = cKDTree(W)
    wt2 = cKDTree(W[:, :2])
    rows = []
    for line in open(exp / a.jsonl):
        r = json.loads(line)
        if r.get("status") != "ok" or "traj_file" not in r:
            rows.append({**{k: r.get(k) for k in ("model", "add_wires", "pair", "status")}})
            continue
        T = densify(np.load(exp / r["traj_file"])[:, :3])
        nb = wt2.query_ball_point(T[:, :2], a.rh)
        above = below = level = 0
        level_dist = []
        for i, idx in enumerate(nb):
            if not idx:
                continue
            zw = W[idx, 2]
            z = T[i, 2]
            if z > zw.max() + a.dz:
                above += 1
            elif z < zw.min() - a.dz:
                below += 1
            else:
                level += 1
                level_dist.append(np.min(np.linalg.norm(W[idx] - T[i], axis=1)))
        n_near = above + below + level
        kind = "電線の所を通らない" if n_near == 0 else ("電線の高さの範囲を通る" if level else ("上を越える" if above and not below else ("下をくぐる" if below and not above else "上と下の両方")))
        rows.append({"model": r["model"], "add_wires": r["add_wires"], "pair": r["pair"], "status": "ok", "kind": kind,
                     "pts_above": above, "pts_below": below, "pts_level": level,
                     "min_dist_wire_m": float(wt3.query(T)[0].min()),
                     "min_dist_at_level_m": float(min(level_dist)) if level_dist else np.nan,
                     "z_min_m": float(T[:, 2].min() - ground), "z_max_m": float(T[:, 2].max() - ground),
                     "len_ratio": r["length"] / r["straight_m"]})
    df = pd.DataFrame(rows)
    out = exp / "splatnav_cross_traj_analysis.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    ok = df[df.status == "ok"]
    print(ok.groupby(["model", "add_wires"]).kind.value_counts().unstack(fill_value=0).to_string())
    print(ok[["model", "add_wires", "pair", "kind", "pts_level", "min_dist_wire_m", "min_dist_at_level_m", "z_min_m", "z_max_m", "len_ratio"]]
          .sort_values(["model", "pair", "add_wires"]).round(2).to_string(index=False))
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

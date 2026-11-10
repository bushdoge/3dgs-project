# フローターの除去方法を実データで比べる（研究メモ §9bc）。
#   none     ：元の地図
#   heur     ：CollisionSplatting（arXiv 2609.35619、github machines-in-motion/CollisionSplatting の scene.py drop_floaters）の経験則
#              ＝ 不透明度 < 0.1 かつ 一番長い軸 > 5 × 一番長い軸の中央値 を捨てる（既定値そのまま。Mip は 3D フィルタ込みの実効値で判定）
#   skyprune ：空の視線で捨てる（sky_prune.py --views even の出力 <model>_skyprune）
# 測るもの（いずれも同じ座標・m）：
#   ・電線の被覆：wire3d.py で復元した電線の点のうち、1σ楕円体（＋2cm）に入る割合
#   ・確実に空いている点の誤占有：奇数番の写真の空の視線で作った点（sky_carve と同じ作り方）のうち、1σ楕円体（＋機体半径 0.25m）に入る割合
#     （skyprune の除去に偶数番を使ったので、評価は奇数番だけにして循環を避ける。0.25m は Splat-Nav 論文の実機の半径）
# 使い方: python3 real/compare_floater_removal.py <exp> <model> [...]  → <exp>/compare_floater_removal.csv
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import splatnav_wires as SW           # noqa: E402
import sky_carve as SC                # noqa: E402
from map_eval import inside_any       # noqa: E402
from real_splatnav import camera_frame   # noqa: E402


def gauss(exp, model, al):
    mu, q, sc, op = SW.load_ply_gt(exp / model / "point_cloud" / "iteration_30000" / "point_cloud.ply", al)
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    scc = np.maximum(sc, SW.MIN_SCALE)
    cov = (R * scc[:, None, :] ** 2) @ R.transpose(0, 2, 1)
    return mu, cov, scc.max(1), op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--kappa", type=float, default=0.25)
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    al, _, _ = camera_frame(rec, 1.4)
    cams = SC.cameras_real(exp, (al, rec), SC.observations(rec))[1::2]
    ca = argparse.Namespace(stride=12, erode=4, wire_dilate=6, bh_thr=12, border=8, obs_dilate=40, close=9,
                            dmin=0.5, dmax=15.0, dstep=0.25, max_points=200000)
    P, _ = SC.carve(cams, ca, np.random.default_rng(0))
    W = np.load(exp / "wire3d_bright.npz")["points"]
    rows = []
    for m in a.models:
        mu, cov, smax, op = gauss(exp, m, al)
        heur = (op < 0.1) & (smax > 5.0 * np.median(smax))
        mu2, cov2, smax2, _ = gauss(exp, f"{m}_skyprune", al)
        for name, (M, Cv, S) in {"none": (mu, cov, smax), "heur": (mu[~heur], cov[~heur], smax[~heur]),
                                  "skyprune": (mu2, cov2, smax2)}.items():
            rows.append({"exp": exp.name, "model": m, "method": name, "n_gauss": len(M),
                         "n_removed": int(len(mu) - len(M)),
                         "wire_cover": float(inside_any(W, M, Cv, S, 1.0, infl2=0.02 ** 2, chunk=5000).mean()),
                         "free_occ_kappa": float(inside_any(P, M, Cv, S, 1.0, infl2=a.kappa ** 2, chunk=5000).mean())})
            print(rows[-1], flush=True)
        # 経験則で捨てたものの性質
        print(f"  {m}: 経験則で捨てた {heur.sum()} 個（一番長い軸の中央値の5倍＝{5*np.median(smax):.3f}m 超・不透明度<0.1）", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(exp / "compare_floater_removal.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

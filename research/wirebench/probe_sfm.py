# 「帯 vs 半平面」（研究メモ §9as）：物体ごとの SfM 点数（初期ガウシアンの数）と消失の関係を見る。
# COLMAP 点群を GT 座標に位置合わせし（wbgeom.align_colmap_to_gt）、各円柱の表面（軸の線分距離 − 半径）から --tol 以内、
# 板は板の面から --tol 以内の点を数える。presence.csv（probe_presence.py）があれば run ごとの保持率と並べ、
# 「SfM 点が 0 の物体ほど消える」かを半径ごとに出す。
# 使い方: python3 probe_sfm.py <exp> [--tol 0.03] [--split train]
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbgeom import align_colmap_to_gt, apply   # noqa: E402


def seg_dist(P, a, b):
    a, b = np.asarray(a), np.asarray(b)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    exp = Path(a.exp)
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    al = align_colmap_to_gt(exp, rec)
    print(f"位置合わせ RMSE {al['rmse']*100:.2f}cm（{al['n_inliers']}/{al['n_cams']} カメラ）")
    P = apply(al, np.array([p.xyz for p in rec.points3D.values()]))
    objs = json.load(open(exp / "gt" / "probe_objects.json"))
    rows = []
    for o in objs:
        if o["kind"] == "cylinder":
            d = seg_dist(P, o["p0"], o["p1"]) - o["radius_m"]
            n = int((np.abs(d) < a.tol).sum())
            rows.append(dict(obj=o["id"], kind="cylinder", radius_mm=round(o["radius_m"] * 1000, 1),
                             tilt=round(o["tilt_deg"], 1), n_sfm=n))
        else:
            C = np.array(o["corners"])
            lo, hi = C.min(0) - a.tol, C.max(0) + a.tol
            hi[2] += o["size_m"][2] + a.tol
            n = int(((P >= lo) & (P <= hi)).all(1).sum())
            rows.append(dict(obj=o["id"], kind="plate", radius_mm=np.nan, tilt=np.nan, n_sfm=n))
    df = pd.DataFrame(rows).set_index("obj")
    pp = exp / "gt" / ("probe_eval_train" if a.split == "train" else "probe_eval") / "presence.csv"
    if pp.exists():
        pr = pd.read_csv(pp, index_col="obj")
        df["n_vanished"] = (pr < 0.5).sum(1)
        df["n_runs"] = pr.shape[1]
    pd.set_option("display.width", 200)
    print(df.sort_values(["kind", "radius_mm", "n_sfm"]).to_string())
    if "n_vanished" in df:
        c = df[df.kind == "cylinder"].assign(has_sfm=lambda t: t.n_sfm > 0)
        print("\n== 半径 × SfM点の有無 ごとの消失（消失 run 数 / run×物体） ==")
        g = c.groupby(["radius_mm", "has_sfm"]).agg(n_obj=("n_sfm", "size"), vanished=("n_vanished", "sum"), runs=("n_runs", "sum"))
        g["rate"] = (g.vanished / g.runs).round(2)
        print(g.to_string())
    df.to_csv(exp / "gt" / "probe_sfm.csv")
    print(f"保存: {exp / 'gt' / 'probe_sfm.csv'}")


if __name__ == "__main__":
    main()

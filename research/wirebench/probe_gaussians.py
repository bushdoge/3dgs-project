# 「帯 vs 半平面」（研究メモ §9as）：描画上で消えた物体は「地図（ガウシアンの集合）」からも消えているか。
# 3DGS を地図として経路計画に使う研究（Splat-Nav 等）は楕円体の幾何で衝突を判定するので、描画では見えなくても
# 透明なガウシアンが物体の位置に残っていれば障害物として扱われうる。学習済み ply を GT 座標に位置合わせし、
# 各円柱の軸から（半径＋--tol）以内にあるガウシアン中心の数と実効不透明度（Mip は 3D フィルタ込み）を数える。
# 対照として同じ円柱を --ctrl-dz だけ上に平行移動した位置（空）でも数える（空にあるフローターの密度の目安）。
# 使い方: python3 probe_gaussians.py <exp> <model_dir名> [...] [--iter 30000] [--tol 0.03]
#   → <exp>/gt/probe_gaussians.csv（モデル×物体）と、保持率（presence.csv）別の要約
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbgeom import align_colmap_to_gt, apply, gaussian_params   # noqa: E402


def seg_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--tol", type=float, default=0.03, help="表面からの距離[m]")
    ap.add_argument("--ctrl-dz", type=float, default=1.0, help="対照（空）の位置：物体を上へ何 m ずらすか")
    a = ap.parse_args()
    exp = Path(a.exp)
    al = align_colmap_to_gt(exp, pycolmap.Reconstruction(str(exp / "sparse" / "0")))
    objs = [o for o in json.load(open(exp / "gt" / "probe_objects.json")) if o["kind"] == "cylinder"]
    pres = None
    pp = exp / "gt" / "probe_eval_train" / "presence.csv"
    if pp.exists():
        pres = pd.read_csv(pp, index_col="obj")
    rows = []
    for m in a.models:
        v = PlyData.read(str(exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply"))["vertex"]
        xyz, op, sc, op_eff, sc_eff, _ = gaussian_params(v)
        P = apply(al, xyz)
        size_w = sc_eff.max(1) * al["s"]   # 最大軸の実効スケール[m]
        for o in objs:
            R = o["radius_m"]
            for kind, dz in (("obj", 0.0), ("ctrl", a.ctrl_dz)):
                p0 = np.array(o["p0"]) + [0, 0, dz]
                p1 = np.array(o["p1"]) + [0, 0, dz]
                sel = seg_dist(P, p0, p1) < R + a.tol
                oe = op_eff[sel]
                rows.append(dict(model=m, obj=o["id"], radius_mm=round(R * 1000, 1), tilt=round(o["tilt_deg"], 1), loc_kind=kind,
                                 n=int(sel.sum()), n_op01=int((oe > 0.1).sum()), n_op05=int((oe > 0.5).sum()),
                                 op_max=float(oe.max()) if sel.any() else 0.0,
                                 op_sum=float(oe.sum()), size_med_mm=float(np.median(size_w[sel]) * 1000) if sel.any() else np.nan,
                                 ret=float(pres.loc[o["id"], m]) if pres is not None and m in pres.columns else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(exp / "gt" / "probe_gaussians.csv", index=False)
    pd.set_option("display.width", 220)
    ob = df[df.loc_kind == "obj"].copy()
    ob["state"] = np.where(ob.ret < 0.5, "消失", "見えている")
    print(f"== 物体の周り（表面から {a.tol*100:.0f}cm 以内）のガウシアン：描画上の状態別（run×物体）==")
    g = ob.groupby(["radius_mm", "state"]).agg(n_runobj=("n", "size"), n_med=("n", "median"), n_op01_med=("n_op01", "median"),
                                                n_op05_med=("n_op05", "median"), zero_any=("n", lambda s: int((s == 0).sum())),
                                                zero_op01=("n_op01", lambda s: int((s == 0).sum())), op_max_med=("op_max", "median"))
    print(g.round(3).to_string())
    ct = df[df.loc_kind == "ctrl"]
    print(f"\n対照（同じ形を {a.ctrl_dz}m 上の空へ）：n 中央値 {ct.n.median():.0f}、不透明度>0.1 の数 中央値 {ct.n_op01.median():.0f}、"
          f"0個の割合 {(ct.n == 0).mean():.2f}")
    van = ob[ob.state == "消失"]
    print("\n消失した run×物体の内訳（n＝全ガウシアン数、n_op01＝不透明度>0.1、op_max＝最大不透明度）:")
    print(van[["model", "obj", "radius_mm", "tilt", "n", "n_op01", "n_op05", "op_max", "ret"]].round(3).to_string(index=False))
    print(f"保存: {exp / 'gt' / 'probe_gaussians.csv'}")


if __name__ == "__main__":
    main()

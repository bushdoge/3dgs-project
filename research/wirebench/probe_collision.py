# 「帯 vs 半平面」第2回（研究メモ §9as）の学習済みモデルを、3DGS を地図にした経路計画（Splat-Nav, arXiv 2403.02751）の
# 衝突判定で見たときに、細い物体が「障害物として地図に載っているか」を測る。画素の誤差（PSNR 等）ではなく地図の幾何で評価する。
# Splat-Nav の衝突幾何：各ガウシアンの γ 信頼楕円体（論文は γ=0.2、χ²₃(0.2)≈1.005＝ほぼ1σ）の和集合。不透明度で除外する記述は無い。
# ここでの判定：円柱の軸を 1cm 間隔で区切り、各点について「1σ楕円体が円柱表面に届いているか」を
#   (x−μ)ᵀ(Σ + R²I)⁻¹(x−μ) ≤ χ²₃(γ)   （Σ は χ² 倍前の共分散、R は円柱半径。楕円体と球のミンコフスキー和の近似）
# で調べ、被覆率（軸のうち障害物に覆われた割合）と最大の隙間[m]を出す。隙間が機体の直径より長ければ計画器はそこを通り抜けうる。
# 不透明度は「全部」（Splat-Nav の記述どおり）と「実効不透明度>0.1」の2通り。対照は同じ円柱を 1m 上の空へずらした位置（空のフローター）。
# 使い方: python3 probe_collision.py <exp> <model_dir名> [...] [--iter 30000] [--gamma 0.2]
#   → <exp>/gt/probe_collision.csv（モデル×物体×位置×不透明度条件）
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from plyfile import PlyData
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbgeom import align_colmap_to_gt, apply, gaussian_params   # noqa: E402
from probe_gaussians import seg_dist                              # noqa: E402

STEP = 0.01   # 軸のサンプル間隔[m]


def quat_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def coverage(axis_pts, mu, cov, R, thr):
    """軸の各点が、どれかの楕円体（Σ+R²I で膨らませたもの）の閾値内にあるか"""
    hit = np.zeros(len(axis_pts), bool)
    if len(mu) == 0:
        return hit
    inv = np.linalg.inv(cov + (R ** 2) * np.eye(3))
    for i in range(0, len(mu), 2000):
        d = axis_pts[None, :, :] - mu[i:i + 2000, None, :]                # (g, a, 3)
        m2 = np.einsum("gai,gij,gaj->ga", d, inv[i:i + 2000], d)
        hit |= (m2 <= thr).any(0)
    return hit


def max_gap(hit, n_step):
    """被覆されていない連続区間の最長[m]"""
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1
        best = max(best, cur)
    return best * n_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--gamma", type=float, default=0.2, help="Splat-Nav の信頼水準 γ（論文の既定 0.2）")
    ap.add_argument("--ctrl-dz", type=float, default=1.0)
    a = ap.parse_args()
    exp = Path(a.exp)
    thr = chi2.ppf(a.gamma, 3)
    al = align_colmap_to_gt(exp, pycolmap.Reconstruction(str(exp / "sparse" / "0")))
    objs = [o for o in json.load(open(exp / "gt" / "probe_objects.json")) if o["kind"] == "cylinder"]
    pp = exp / "gt" / "probe_eval_train" / "presence.csv"
    pres = pd.read_csv(pp, index_col="obj") if pp.exists() else None
    rows = []
    for m in a.models:
        v = PlyData.read(str(exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply"))["vertex"]
        xyz, _, _, op_eff, sc_eff, _ = gaussian_params(v)
        Rq = quat_to_R(np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64))
        mu = apply(al, xyz)
        Rw = al["R"][None] @ Rq                                   # GT 座標での回転
        sw = sc_eff * al["s"]                                     # GT 座標でのスケール[m]
        reach = np.sqrt(thr) * sw.max(1)                          # 楕円体の最大半径
        for o in objs:
            R = o["radius_m"]
            for kind, dz in (("obj", 0.0), ("ctrl", a.ctrl_dz)):
                p0 = np.array(o["p0"]) + [0, 0, dz]
                p1 = np.array(o["p1"]) + [0, 0, dz]
                n = max(2, int(round(np.linalg.norm(p1 - p0) / STEP)) + 1)
                ax = p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)
                near = seg_dist(mu, p0, p1) < R + reach + 1e-3
                for opk, sel in (("all", near), ("op01", near & (op_eff > 0.1))):
                    idx = np.nonzero(sel)[0]
                    cov = (Rw[idx] * sw[idx][:, None, :] ** 2) @ Rw[idx].transpose(0, 2, 1)
                    hit = coverage(ax, mu[idx], cov, R, thr)
                    rows.append(dict(model=m, obj=o["id"], radius_mm=round(R * 1000, 1), tilt=round(o["tilt_deg"], 1),
                                     loc_kind=kind, op=opk, n_g=len(idx), cover=float(hit.mean()),
                                     max_gap_m=max_gap(hit, np.linalg.norm(p1 - p0) / (n - 1)),
                                     ret=float(pres.loc[o["id"], m]) if pres is not None and m in pres.columns else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(exp / "gt" / "probe_collision.csv", index=False)
    pd.set_option("display.width", 220)
    ob = df[df.loc_kind == "obj"].copy()
    ob["state"] = np.where(ob.ret < 0.5, "消失", "見えている")
    print(f"== Splat-Nav 型の衝突幾何（γ={a.gamma}）で見た被覆率と最大の隙間（run×物体の中央値・件数）==")
    g = ob.groupby(["op", "radius_mm", "state"]).agg(n=("cover", "size"), cover_med=("cover", "median"),
                                                    cover0=("cover", lambda s: int((s == 0).sum())),
                                                    full=("cover", lambda s: int((s >= 0.99).sum())),
                                                    gap_med=("max_gap_m", "median"),
                                                    gap_gt30cm=("max_gap_m", lambda s: int((s > 0.3).sum())))
    print(g.round(3).to_string())
    ct = df[df.loc_kind == "ctrl"]
    print("\n対照（1m 上の空）：被覆率の中央値・被覆>0 の割合")
    print(ct.groupby("op").agg(cover_med=("cover", "median"), any_hit=("cover", lambda s: float((s > 0).mean()))).round(3).to_string())
    print(f"保存: {exp / 'gt' / 'probe_collision.csv'}")


if __name__ == "__main__":
    main()

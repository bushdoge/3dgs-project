# 「帯 vs 半平面」（研究メモ §9as）の不確かさ：視点ブートストラップで超過比 X の 95% 区間と、向き（傾き）別の R を出す。
# collect_probe.py と同じ定義（e＝Σsse/Σ境界長、R＝e_円柱/e_板、X＝R/R_blur(σ_fit)）を、視点を復元抽出して B 回計算する。
# さらに物体単位の X（物体ごとに σ_fit のぼかし基準で割る）を出し、太い側（≥60mm＝2本の境界が独立）の物体×run を
# 帰無分布として細い側と比べる。run 間で同じ物体は相関するので、run 平均した物体単位（n＝物体数）の比較も出す。
# 使い方: python3 probe_bootstrap.py <exp> [--split train|test] [--B 1000] [--exclude 13]（消失した物体を除く）
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# probe_eval*/ にある集計側の出力（評価 CSV ではないので読まない）
AUX_CSV = {"summary.csv", "bootstrap_X.csv", "per_object_X.csv", "presence.csv"}


def x_table(df, models):
    """視点集合 df からモデル×半径の X を返す（collect_probe.py と同じ手順）"""
    g = df.groupby(["model", "kind", "radius_mm"]).agg(sse=("sse", "sum"), blen=("bnd_len_px", "sum")).reset_index()
    g["e"] = g.sse / g.blen
    plate = g[g.kind == "plate"].set_index("model").e
    cyl = g[g.kind == "cylinder"].copy()
    cyl["R"] = cyl.e / cyl.model.map(plate)
    blur = cyl[cyl.model.str.startswith("blur_s")].copy()
    blur["sigma"] = blur.model.str[6:].astype(float)
    sig = np.sort(blur.sigma.unique())
    pe = np.array([plate[f"blur_s{s:g}"] for s in sig])
    out = {}
    for m in models:
        s_fit = float(np.interp(np.log(plate[m]), np.log(pe), sig))
        t = cyl[cyl.model == m]
        for _, r in t.iterrows():
            rb = blur[blur.radius_mm == r.radius_mm].sort_values("sigma")
            out[(m, r.radius_mm)] = r.R / float(np.interp(s_fit, rb.sigma, rb.R))
    return pd.Series(out)


def per_object(df, models, d, exclude):
    """物体単位の X と、太い側（≥60mm）を帰無にした比較"""
    from scipy import stats
    g = df.groupby(["model", "obj", "kind", "radius_mm", "tilt_deg"], dropna=False).agg(
        sse=("sse", "sum"), bl=("bnd_len_px", "sum")).reset_index()
    g["e"] = g.sse / g.bl
    plate = g[g.kind == "plate"].set_index("model").e
    c = g[g.kind == "cylinder"].copy()
    c["R"] = c.e / c.model.map(plate)
    blur = c[c.model.str.startswith("blur_s")].copy()
    blur["sigma"] = blur.model.str[6:].astype(float)
    sig = np.sort(blur.sigma.unique())
    pe = np.array([plate[f"blur_s{s:g}"] for s in sig])
    rows = []
    for m in models:
        s_fit = float(np.interp(np.log(plate[m]), np.log(pe), sig))
        for _, r in c[c.model == m].iterrows():
            b = blur[blur.obj == r.obj].sort_values("sigma")
            rows.append(dict(model=m, obj=r.obj, radius_mm=r.radius_mm, tilt=r.tilt_deg, R=r.R,
                             X=r.R / float(np.interp(s_fit, b.sigma, b.R))))
    X = pd.DataFrame(rows)
    X.to_csv(d / "per_object_X.csv", index=False)
    print("\n== 物体単位の X（行＝半径・傾き） ==")
    print(X.pivot_table(index=["radius_mm", "tilt"], columns="model", values="X").round(2).to_string())
    null = X[X.radius_mm >= 60]
    nobj = null.groupby("obj").X.apply(lambda v: np.exp(np.log(v).mean()))
    print(f"\n帰無（≥60mm）物体×run n={len(null)}: 範囲 {null.X.min():.2f}〜{null.X.max():.2f}、95%点 {null.X.quantile(.95):.2f}"
          f"／run 幾何平均の物体単位 n={len(nobj)}: {nobj.min():.2f}〜{nobj.max():.2f}")
    for rad in sorted(X.radius_mm.unique()):
        if rad >= 60:
            continue
        t = X[(X.radius_mm == rad) & ~X.obj.isin(exclude)]
        to = t.groupby("obj").X.apply(lambda v: np.exp(np.log(v).mean()))
        u = stats.mannwhitneyu(np.log(t.X), np.log(null.X), alternative="greater").pvalue
        uo = stats.mannwhitneyu(np.log(to), np.log(nobj), alternative="greater").pvalue
        print(f"{rad:5g}mm（除外 {exclude}）物体×run n={len(t)} 中央値 {t.X.median():.2f}・MWU p={u:.4f}（run を独立扱い）"
              f"｜物体単位 n={len(to)} {np.round(to.values, 2).tolist()}・MWU p={uo:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--split", choices=["test", "train"], default="train")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--exclude", type=int, nargs="*", default=[], help="物体単位の比較から除く物体 id（消失したもの）")
    a = ap.parse_args()
    d = Path(a.exp) / "gt" / ("probe_eval" if a.split == "test" else "probe_eval_train")
    df = pd.concat([pd.read_csv(p) for p in sorted(d.glob("*.csv")) if p.name not in AUX_CSV], ignore_index=True)
    models = sorted(m for m in df.model.unique() if not m.startswith("blur_s"))
    views = np.array(sorted(df.view.unique()))
    by_view = {v: g for v, g in df.groupby("view")}
    point = x_table(df, models)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.B):
        pick = rng.choice(views, len(views), replace=True)
        boots.append(x_table(pd.concat([by_view[v] for v in pick]), models))
    bt = pd.concat(boots, axis=1)
    res = pd.DataFrame({"X": point, "lo": bt.quantile(0.025, axis=1), "hi": bt.quantile(0.975, axis=1)})
    res.index.names = ["model", "radius_mm"]
    pd.set_option("display.width", 250)
    print(f"== X と視点ブートストラップ 95% 区間（{a.split}、{len(views)} 視点、B={a.B}） ==")
    print(res.round(2).reset_index().assign(s=lambda r: r.X.round(2).astype(str) + " [" + r.lo.round(2).astype(str)
          + ", " + r.hi.round(2).astype(str) + "]").pivot(index="radius_mm", columns="model", values="s").to_string())
    # 向き別の R（板比）。向きで結論が変わらないかの確認
    g = df[df.kind == "cylinder"].groupby(["model", "radius_mm", "tilt_deg"]).agg(sse=("sse", "sum"), blen=("bnd_len_px", "sum"))
    pl = df[df.kind == "plate"].groupby("model")[["sse", "bnd_len_px"]].sum().pipe(lambda t: t.sse / t.bnd_len_px)
    g["R"] = (g.sse / g.blen) / pl.reindex(g.index.get_level_values(0)).values
    print("\n== 向き別 R（板比。列＝傾き°） ==")
    print(g.R.unstack("tilt_deg").round(2).loc[models + [m for m in ["blur_s0.5", "blur_s1"] if m in pl.index]].to_string())
    res.to_csv(d / "bootstrap_X.csv")
    per_object(df, models, d, a.exclude)
    print(f"\n保存: {d / 'bootstrap_X.csv'}")


if __name__ == "__main__":
    sys.exit(main())

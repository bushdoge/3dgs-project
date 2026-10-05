# 「帯 vs 半平面」検定（研究メモ §9as）の集計と判定。eval_probe.py の CSV（<exp>/gt/probe_eval/*.csv）を読み、
#   (1) 物体ごとの誤差密度 e（全テスト視点で Σsse / Σ境界長）と、同じモデルの板（半平面）に対する比 R = e_円柱 / e_板
#   (2) ぼかし基準：GT をガウシアン σ でぼかしただけの擬似モデルの R_blur(σ)。線形なぼかしでも細い帯では R>1 になる
#   (3) モデルごとに「板の e が同じになる σ」を当てはめ（σ_fit）、超過比 X = R_model / R_blur(σ_fit) を出す
# 判定（設計書 §6 をぼかし基準で補正したもの）:
#   H_band：細い帯（見かけ幅 ≲3px）で X が 1 を run 間ばらつきの外まで超え、太い側（≳12px）で X≈1
#   H_indep：全幅で X≈1（細い帯の誤差は「ぼかしが2本の境界に掛かった分」で説明できる）
# 使い方: python3 collect_probe.py <exp> [--iter 30000] [--split train]
#   → <exp>/gt/probe_eval/summary.csv（モデル×半径の表）と figs/probe_band_vs_halfplane.png（このdir）
#   --split train は probe_eval_train/ を読み、図は probe_band_vs_halfplane_train.png
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# probe_eval*/ にある集計側の出力（評価 CSV ではないので読まない）
AUX_CSV = {"summary.csv", "bootstrap_X.csv", "per_object_X.csv", "presence.csv"}

HERE = Path(__file__).parent


def per_object(df):
    """モデル×物体で視点を貯めた誤差密度（Σsse/Σ境界長）と見かけ幅の中央値"""
    g = df.groupby(["model", "obj", "kind", "radius_mm", "tilt_deg"], dropna=False)
    out = g.agg(sse=("sse", "sum"), blen=("bnd_len_px", "sum"), n_views=("view", "nunique"),
                width_px=("width_px", "median")).reset_index()
    out["e"] = out["sse"] / out["blen"]
    return out


def ratio_table(po):
    """半径ごと（向き3種を貯める）の e と、板に対する比 R"""
    cyl = po[po.kind == "cylinder"].groupby(["model", "radius_mm"]).agg(
        sse=("sse", "sum"), blen=("blen", "sum"), width_px=("width_px", "median")).reset_index()
    cyl["e"] = cyl["sse"] / cyl["blen"]
    plate = po[po.kind == "plate"].set_index("model")["e"]
    cyl["e_plate"] = cyl["model"].map(plate)
    cyl["R"] = cyl["e"] / cyl["e_plate"]
    return cyl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    a = ap.parse_args()
    d = Path(a.exp) / "gt" / ("probe_eval" if a.split == "test" else "probe_eval_train")
    df = pd.concat([pd.read_csv(p) for p in sorted(d.glob("*.csv")) if p.name not in AUX_CSV], ignore_index=True)
    po = per_object(df)
    rt = ratio_table(po)

    blur = rt[rt.model.str.startswith("blur_s")].copy()
    blur["sigma"] = blur.model.str.extract(r"blur_s([\d.]+)")[0].astype(float)
    sig = np.sort(blur.sigma.unique())
    plate_e_blur = np.array([blur[blur.sigma == s].e_plate.iloc[0] for s in sig])
    models = [m for m in rt.model.unique() if not m.startswith("blur_s")]

    rows = []
    for m in models:
        t = rt[rt.model == m]
        ep = t.e_plate.iloc[0]
        # 板の e が単調に増える範囲で log e を σ について線形補間して σ_fit を求める
        s_fit = float(np.interp(np.log(ep), np.log(plate_e_blur), sig))
        clipped = ep < plate_e_blur.min() or ep > plate_e_blur.max()
        for _, r in t.iterrows():
            rb = blur[blur.radius_mm == r.radius_mm].sort_values("sigma")
            R_blur = float(np.interp(s_fit, rb.sigma, rb.R))
            rows.append({"model": m, "radius_mm": r.radius_mm, "width_px": round(r.width_px, 2),
                         "e": r.e, "e_plate": ep, "R": r.R, "sigma_fit": round(s_fit, 3),
                         "sigma_clipped": clipped, "R_blur": R_blur, "X": r.R / R_blur})
    for _, r in blur.iterrows():
        rows.append({"model": r.model, "radius_mm": r.radius_mm, "width_px": round(r.width_px, 2), "e": r.e,
                     "e_plate": r.e_plate, "R": r.R, "sigma_fit": r.sigma, "sigma_clipped": False,
                     "R_blur": r.R, "X": 1.0})
    S = pd.DataFrame(rows)
    S.to_csv(d / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("== ぼかし基準 R_blur（板に対する比。列＝σ） ==")
    print(blur.pivot(index="radius_mm", columns="sigma", values="R").round(3).to_string())
    if models:
        print("\n== モデル：R（板比）と X（ぼかし基準に対する超過比） ==")
        sm = S[S.model.isin(models)]
        print(sm.pivot(index="radius_mm", columns="model", values="R").round(3).to_string())
        print()
        print(sm.pivot(index="radius_mm", columns="model", values="X").round(3).to_string())
        print("\nσ_fit:", {m: S[S.model == m].sigma_fit.iloc[0] for m in models},
              "（σ_clipped=True は当てはめ範囲外）" if S[S.model.isin(models)].sigma_clipped.any() else "")
        # 手法ごと（seed をまとめて）細い帯（≤3px）と太い側（≥12px）の X
        kind = lambda m: re.sub(r"_r\d+.*$", "", m)
        sm = sm.assign(method=sm.model.map(kind))
        for meth, g in sm.groupby("method"):
            thin = g[g.width_px <= 3.5].groupby("model").X.mean()
            thick = g[g.width_px >= 11].groupby("model").X.mean()
            print(f"{meth:16s} 細い帯(≤3.5px) X = {thin.round(3).tolist()}  太い側(≥11px) X = {thick.round(3).tolist()}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for s in sig:
            b = blur[blur.sigma == s].sort_values("width_px")
            ax[0].plot(b.width_px, b.R, color="0.6", lw=1, ls="--")
            ax[0].annotate(f"σ={s:g}", (b.width_px.iloc[0], b.R.iloc[0]), fontsize=7, color="0.4")
        for m in models:
            t = S[S.model == m].sort_values("width_px")
            ax[0].plot(t.width_px, t.R, marker="o", label=m)
            ax[1].plot(t.width_px, t.X, marker="o", label=m)
        for x in ax:
            x.set_xscale("log")
            x.set_yscale("log")   # 消失した物体（§9as：R≈10〜70）があっても太い側が読めるように
            x.axhline(1, color="k", lw=0.8)
            x.set_xlabel(f"apparent width [px] (median over {a.split} views)")
        ax[0].set_ylabel("R = e_cylinder / e_plate (per boundary)")
        ax[0].set_title("per-boundary error vs half-plane (grey: GT blurred by σ)")
        ax[1].set_ylabel("X = R_model / R_blur(σ_fit)")
        ax[1].set_title("excess over the blur-only reference")
        if models:
            ax[1].legend(fontsize=7)
        fig.tight_layout()
        (HERE / "figs").mkdir(exist_ok=True)
        fn = "probe_band_vs_halfplane" + ("" if a.split == "test" else "_train") + ".png"
        fig.savefig(HERE / "figs" / fn, dpi=130)
        print(f"\n図: {HERE / 'figs' / fn}")
    except Exception as e:
        print("図の作成に失敗:", e)
    print(f"保存: {d / 'summary.csv'}")


if __name__ == "__main__":
    main()

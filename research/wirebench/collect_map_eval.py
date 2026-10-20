# 地図評価（map_eval.py、研究メモ §9at）の結果を全シーン分集め、各 run の電線のみ PSNR（wire_eval の summary.json、md2）と並べる。
# 手法（モデル名から推定）ごとの平均と、PSNR と地図指標の順位相関（Spearman）を出す。
# 使い方: python3 collect_map_eval.py  → research/wirebench/map_eval_results.csv
import glob
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/workspace/experiments")


def method_of(m):
    m = m.replace("output_", "")
    m = re.sub(r"(_r\d+|_rerun)$", "", m)
    m = re.sub(r"_?30k|_?30000", "", m)
    return m or "3dgs"


def psnr_of(exp, m):
    f = ROOT / exp / "gt" / f"wire_eval_{m}_30000" / "summary.json"
    return json.load(open(f)).get("psnr_wireonly") if f.exists() else None


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "*wb_*" / "gt" / "map_eval.csv")))], ignore_index=True)
    d["method"] = d.model.map(method_of)
    d["psnr_wireonly"] = [psnr_of(e, m) for e, m in zip(d.exp, d.model)]
    out = Path(__file__).resolve().parent / "map_eval_results.csv"
    d.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["psnr_wireonly", "wire_cover", "wire_cover_op01", "wire_ngap30_op01", "false_occ_shell", "false_occ_shell_kappa",
            "false_occ_box_kappa"]
    g = d.groupby(["exp", "method"]).agg(n=("model", "size"), **{c: (c, "mean") for c in cols})
    print(g.round(4).to_string())
    for exp in ["20260704_wb_r15mm_f80_open"]:
        s = g.loc[exp]
        s = s[s.n >= 3]
        print(f"\n{exp}（3run 以上の手法）PSNR と地図指標の順位相関:")
        for c in cols[1:]:
            r, p = spearmanr(s.psnr_wireonly, s[c])
            print(f"  {c:24s} ρ={r:+.2f}  p={p:.2f}")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

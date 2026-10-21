# sky_carve.py（研究メモ §9aw）の結果を集め、シーン×手法×不透明度の足切りごとに平均する。
# 使い方: python3 collect_sky_carve.py  → research/wirebench/sky_carve_results.csv と要約の表示
import glob
from pathlib import Path

import pandas as pd

from collect_map_eval import method_of

ROOT = Path("/workspace/experiments")


def main():
    fs = sorted(glob.glob(str(ROOT / "*" / "sky_carve.csv")) + glob.glob(str(ROOT / "*" / "gt" / "sky_carve.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["method"] = d.model.map(method_of)
    out = Path(__file__).resolve().parent / "sky_carve_results.csv"
    d.to_csv(out, index=False)
    cols = ["n_points", "occ_ellip", "occ_aabb", "occ_ellip_h-99_2", "occ_ellip_h2_5", "occ_ellip_h5_10", "astar_ok_ellip", "astar_ok_aabb"]
    g = d.groupby(["exp", "method", "opacity_thr"]).agg(n_maps=("model", "size"), **{c: (c, "mean") for c in cols},
                                                         astar_aabb_min=("astar_ok_aabb", "min"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(g.round(3).to_string())
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

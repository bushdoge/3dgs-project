# splatnav_wires.py（研究メモ §9au）の結果を全シーン分集め、手法×不透明度閾値ごとに
# 計画の成功率・電線への衝突率・経路長の GT 地図比・占有ボクセル率をまとめる。
# 使い方: python3 collect_splatnav.py  → research/wirebench/splatnav_results.csv（1行＝モデル×閾値×組）と要約の表示
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from collect_map_eval import method_of

ROOT = Path("/workspace/experiments")


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "*" / "gt" / "splatnav_wires.csv")))], ignore_index=True)
    d["method"] = np.where(d.model == "gt_wires", "GT", d.model.map(method_of))
    d["opacity_thr"] = d.opacity_thr.fillna(-1)
    d["ok"] = d.status == "ok"
    d["astar_fail"] = d.status.str.startswith("fail") & d.length.isna()
    out = Path(__file__).resolve().parent / "splatnav_results.csv"
    d.to_csv(out, index=False)
    g = d.groupby(["exp", "method", "opacity_thr"]).agg(
        n_maps=("model", "nunique"), n=("pair", "size"), occ_voxel=("occ_voxel", "mean"),
        success=("ok", "mean"), collide=("collide", lambda s: float((s == True).sum())),
        len_ratio_med=("len_ratio", "median"), clear_min_cm=("wire_clear_m", lambda s: s.min() * 100))
    g["collide_rate"] = g.collide / (g.success * g.n).replace(0, np.nan)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(g.round(3).to_string())
    print(f"保存: {out}")


if __name__ == "__main__":
    main()

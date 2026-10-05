# 選択性（細線が他のエッジより悪い：§0-2・§9y-2・§9ar-5）が「学習視点でも出るか」を測る（研究メモ §9as の妥当性確認）。
# 学習視点で出る＝細線は「表現（当てはめ）」の段階で既に選択的に悪い → 帯vs半平面を学習視点で検定しても当初の問いに忠実。
# 学習視点で出ない＝選択性は新しい視点への汎化で生じる → 検定の設計を考え直す。
# 床なしGT（C3）の 3px / 1.5px × 素3DGS 7k・30k・Mip 30k × 3run を、test と train の両方で
#   eval_contrast_regions.py（thin − edge_other、md2）と analyze_selectivity.py（thin / edge_sil / edge_tex、DIL=0）で測る。
# 前提：各runの train/ours_<it>/ をレンダ済み（tmp/c3_train_render.sh）。
# 使い方: python3 selectivity_split.py → research/wirebench/selectivity_split.csv
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPS = {"3px": "20260924_wb_r15mm_f80_open_clean", "1.5px": "20260924_wb_r15mm_f80_open_res512_clean"}
MODELS = [("3dgs7k", "output_3dgs7k_r{r}", 7000), ("3dgs30k", "output_3dgs30k_r{r}", 30000), ("mip30k", "output_mip30k_r{r}", 30000)]


def run(script, exp, it, model, split, extra_env):
    env = dict(os.environ, SPLIT=split, **extra_env)
    out = subprocess.run(["python3", str(HERE / script), f"/workspace/experiments/{exp}", str(it), model],
                         capture_output=True, text=True, env=env).stdout
    return out


def region_psnr(out, names):
    """表の行頭が領域名の行から最初の数値（PSNR）を拾う"""
    res = {}
    for n in names:
        m = re.search(rf"^{n}\s+([-\d.]+)", out, re.M)
        res[n] = float(m.group(1)) if m else float("nan")
    return res


def main():
    rows = []
    for width, exp in EXPS.items():
        for meth, pat, it in MODELS:
            for r in (1, 2, 3):
                model = pat.format(r=r)
                for split in ("test", "train"):
                    a = region_psnr(run("eval_contrast_regions.py", exp, it, model, split, {}), ["thin", "edge_other"])
                    b = region_psnr(run("analyze_selectivity.py", exp, it, model, split, {"DIL": "0"}), ["thin", "edge_sil", "edge_tex"])
                    rows.append(dict(width=width, method=meth, run=r, split=split,
                                     thin_md2=a["thin"], edge_other=a["edge_other"], sel_other=a["thin"] - a["edge_other"],
                                     thin_md0=b["thin"], edge_sil=b["edge_sil"], edge_tex=b["edge_tex"], sel_sil=b["thin"] - b["edge_sil"]))
                    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "selectivity_split.csv", index=False)
    pd.set_option("display.width", 220)
    g = df.groupby(["width", "method", "split"])[["thin_md2", "edge_other", "sel_other", "thin_md0", "edge_sil", "sel_sil"]].mean().round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()

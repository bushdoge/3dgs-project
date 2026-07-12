# Mipスイープ（Phase 1-7）の結果を実験dirの summary.json / sfm_analysis.json から集めて
# mip_sweep_results.csv に出力する。第1波の素3DGS値も同じ形式で再掲し、1ファイルで引けるようにする。
# 行 = 実験×手法×run。アンカー条件（r15mm open）のみ run2（分散見積もり）がある。
# 使い方: python3 collect_mip_sweep.py   （/workspace で実行）

import csv
import json
import re
from pathlib import Path

E = Path("/workspace/experiments")
OUT = Path(__file__).parent / "mip_sweep_results.csv"

EXPS = [f"20260704_wb_r{r}mm_f80_{v}" for r in ["08", "15", "30", "60"]
        for v in ["open", "enclosed"]]
# (実験dir, 手法, run, 評価dir名)
ROWS = [(e, "3dgs", 1, "wire_eval_7000") for e in EXPS] + \
       [(e, "mip", 1, "wire_eval_output_mip_7000") for e in EXPS] + \
       [("20260704_wb_r15mm_f80_open", "3dgs", 2, "wire_eval_output_rerun_7000"),
        ("20260704_wb_r15mm_f80_open", "mip", 2, "wire_eval_output_mip_rerun_7000")]

COLS = ["exp", "wire_radius_m", "frames", "variant", "method", "run", "iters",
        "sfm_points", "wire_span_pts_3cm", "thin_px_share_pct", "align_rmse_m",
        "psnr_all", "psnr_wire", "psnr_bg", "gap_db", "psnr_wire_min",
        "disappearance_rate", "recall_ren_worst"]

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for exp, method, run, ev in ROWS:
        m = re.match(r"\d+_wb_r(\d+)mm_f(\d+)_(\w+)", exp)
        radius, frames, variant = int(m[1]) / 1000, int(m[2]), m[3]
        s = json.load(open(E / exp / "gt" / ev / "summary.json"))
        sfm = json.load(open(E / exp / "gt" / "sfm_analysis.json"))
        w.writerow([exp, radius, frames, variant, method, run, s["iteration"],
                    sfm["n_points3d"], sfm["wire_span_points"]["3cm"],
                    round(sfm["thin_pixel_share_pct"], 3), round(sfm["align_rmse_m"], 4),
                    round(s["psnr_all"], 2), round(s["psnr_wire"], 2),
                    round(s["psnr_bg"], 2), round(s["gap_db"], 2),
                    round(s["psnr_wire_min"], 2),
                    round(s["disappearance_rate"], 4), round(s["recall_ren_worst"], 3)])
print(f"書き出し: {OUT}（{len(ROWS)}行）")

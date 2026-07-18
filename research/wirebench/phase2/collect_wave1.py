# Phase 2 の結果集計：第1波（裾ロス介入8本）＋第2波（SS教師・30k統制）＋ベースラインを1枚の表にする
# 出典: gt/wire_eval_<model>_<iter>{,_md0}/summary.json・error_budget_<model>_<iter>.json・tail_map_<model>_<iter>.json
# 使い方: python3 collect_wave1.py [実験dir]（既定はアンカー）→ 標準出力＋ phase2_wave1_results.csv

import csv
import json
import sys
from pathlib import Path

exp = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/workspace/experiments/20260704_wb_r15mm_f80_open")
models = ["output", "output_mip", "output_mip_w3", "output_mip_w10", "output_mip_w30",
          "output_mip_adg1", "output_mip_adg2", "output_mip_tk1", "output_mip_tk5",
          "output_mip_vad",
          "output_mip_ss2_f2x", "output_mip_ss2_f1x", ("output_mip_30k", 30000),
          ("output_mip_30k_rerun", 30000), ("output_mip_w3_30k", 30000),
          ("output_mip_w3_30k_rerun", 30000), ("output_mip_adg2_30k", 30000),
          ("output_mip_30k_ks005", 30000), ("output_mip_30k_ks005_rerun", 30000),
          ("output_mip_30k_ks0025", 30000), ("output_mip_30k_ks03", 30000),
          ("output_mip_ctr002_30k", 30000), ("output_mip_ctr02_30k", 30000),
          ("output_mip_ctr0002_30k", 30000)]

rows = []
for m in models:
    m, it = m if isinstance(m, tuple) else (m, 7000)
    we = f"wire_eval_{it}" if m == "output" else f"wire_eval_{m}_{it}"   # 既定outputはdir名にモデル名が付かない
    try:
        s = json.load(open(exp / "gt" / we / "summary.json"))
    except FileNotFoundError:
        continue
    row = {"model": m, "psnr_all": round(s["psnr_all"], 2),
           "psnr_wire": round(s["psnr_wire"], 2), "psnr_bg": round(s["psnr_bg"], 2),
           "gap_db": round(s["gap_db"], 2), "psnr_wire_min": round(s["psnr_wire_min"], 2),
           "disappear": round(s["disappearance_rate"], 4),
           "recall_worst": round(s["recall_ren_worst"], 3)}
    try:
        s0 = json.load(open(exp / "gt" / f"{we}_md0" / "summary.json"))
        row["psnr_wire_md0"] = round(s0["psnr_wire"], 2)
        row["gap_db_md0"] = round(s0["gap_db"], 2)
    except FileNotFoundError:
        row["psnr_wire_md0"] = row["gap_db_md0"] = None
    try:
        eb = json.load(open(exp / "gt" / f"error_budget_{m}_{it}.json"))
        row["top5_share"] = eb["top5pct_sections_err_share"]
        row["pos_share"] = eb["position_share_of_luma_section_err"]
    except FileNotFoundError:
        row["top5_share"] = row["pos_share"] = None
    try:
        tm = json.load(open(exp / "gt" / f"tail_map_{m}_{it}.json"))
        row["worst_view_err_share"] = max(pv["err_share"] for pv in tm["per_view"])
    except FileNotFoundError:
        row["worst_view_err_share"] = None
    rows.append(row)

cols = ["model", "psnr_all", "psnr_wire", "psnr_bg", "gap_db", "psnr_wire_min",
        "psnr_wire_md0", "gap_db_md0", "disappear", "recall_worst",
        "top5_share", "pos_share", "worst_view_err_share"]
out = Path(__file__).parent / "phase2_wave1_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

fmt = "{:<16}" + "{:>10}" * (len(cols) - 1)
print(fmt.format(*cols))
for r in rows:
    print(fmt.format(*[str(r[c]) if r[c] is not None else "-" for c in cols]))
print(f"保存: {out}")

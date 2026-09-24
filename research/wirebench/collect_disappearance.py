# 消失率を「床（測れる上限）」付きで全条件ぶん集計する（監査R-09・修正計画B4）。
# eval_wires.py（2026-09-24版）を各 run に再実行し、summary.json の disappearance_by_class から
#   消失率・床（細線を inpaint で消した GT でも検出されてしまう割合から決まる上限）・正規化消失率
# を 混合（電線＋棒）/ 電線のみ / 棒のみ で disappearance_results.csv に出す。
# 既存キーの値は変わらない（eval_wires.py は追記のみ）ので、実験dirで直接再実行してよい。
# 使い方: python3 collect_disappearance.py [--no-eval]   （--no-eval は summary が新形式のものだけ集計）
import csv
import json
import subprocess
import sys
from pathlib import Path

E = Path("/workspace/experiments")
HERE = Path(__file__).parent

RUNS = []
# §9b / §9w-2 の第1波（太さ×背景・密視点）
for R in ("08", "15", "30", "60"):
    for V in ("open", "enclosed"):
        RUNS += [(f"20260704_wb_r{R}mm_f80_{V}", 7000, "output"), (f"20260704_wb_r{R}mm_f80_{V}", 7000, "output_mip")]
A = "20260704_wb_r15mm_f80_open"
RUNS += [(A, 7000, "output_absgs"), (A, 7000, "output_2dgs"), (A, 30000, "output_raings_30k"),
         (A, 7000, "output_sadgs_7000"), (A, 30000, "output_sadgs_30000"), (A, 30000, "output_3dgs30k_r1"),
         (A, 30000, "output_mip_30k"), (A, 30000, "output_mip_30k_ks005")]
# 疎視点（§9f・§9r・§9s）とシーン変種（§9s）
F = "20260705_wb_r15mm_f40_open"
RUNS += [(F, 7000, m) for m in ("output", "output_rerun", "output_mip", "output_mip_rerun", "output_mip_np1", "output_mip_np1_rerun")]
RUNS += [(F, 30000, m) for m in ("output_mip_30k", "output_mip_30k_rerun", "output_mip_np1_30k", "output_mip_np1_30k_rerun")]
RUNS += [("20260705_wb_r15mm_f40_enclosed", 7000, m) for m in ("output", "output_mip")]
for L in ("Lstreet", "Lcrossing"):
    e = f"20260726_wb_r15mm_f40_open_{L}"
    RUNS += [(e, 7000, "output")] + [(e, 30000, m) for m in ("output_30k", "output_mip_30k", "output_mip_30k_rerun",
                                                             "output_mip_np1_30k", "output_mip_np1_30k_rerun")]
    e = f"20260726_wb_r15mm_f80_open_{L}"
    RUNS += [(e, 7000, "output"), (e, 30000, "output_30k"), (e, 30000, "output_mip_30k_ks005")]

no_eval = "--no-eval" in sys.argv
rows = []
for exp, it, model in RUNS:
    if not (E / exp / model / "test" / f"ours_{it}").exists():
        print(f"skip（出力なし）: {exp}/{model}")
        continue
    d = E / exp / "gt" / (f"wire_eval_{it}" if model == "output" else f"wire_eval_{model}_{it}")
    f = d / "summary.json"
    if not no_eval and (not f.exists() or "disappearance_by_class" not in json.load(open(f))):
        subprocess.run([sys.executable, str(HERE / "eval_wires.py"), str(E / exp), str(it), model],
                       check=True, capture_output=True)
    s = json.load(open(f))
    if "disappearance_by_class" not in s:
        continue
    row = {"exp": exp, "model_dir": model, "iters": it, "n_views": s["n_test_views"],
           "disappearance_rate_legacy": round(s["disappearance_rate"], 4)}
    for c in ("mix", "wire", "bar"):
        dc = s["disappearance_by_class"].get(c) or {}
        for k in ("disappearance_rate", "max_measurable_disappearance", "disappearance_norm"):
            v = dc.get(k)
            row[f"{c}_{k}"] = "" if v is None or v != v else round(v, 4)
    rows.append(row)
    print(f"{exp[-28:]:28s} {model:26s} 混合 {row['mix_disappearance_rate']!s:>7} (上限 {row['mix_max_measurable_disappearance']!s:>6}, 正規化 {row['mix_disappearance_norm']!s:>7})"
          f"  電線 {row['wire_disappearance_rate']!s:>7} (上限 {row['wire_max_measurable_disappearance']!s:>6}, 正規化 {row['wire_disappearance_norm']!s:>7})")

with open(HERE / "disappearance_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"saved: {HERE / 'disappearance_results.csv'}（{len(rows)}行）")

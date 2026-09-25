# 手法横断ベンチ（§9af、監査R-17 → 修正計画C4）の30k・各3run以上を集めて method_bench_results.csv を作り直す。
# 指標は電線のみPSNR（psnr_wireonly、md2）を本命にし、md0・棒・背景・床付き消失率を併記する。
# 旧版（7k/30k 混在・各1run・電線＋棒の混合指標）は git 履歴（e44cfbf）と研究メモ §9af に残っている。
# summary が無い（または電線のみ指標が無い）run は eval_wires.py をその場で実行する。
# 使い方: python3 collect_method_bench.py
#   → method_bench_results.csv（1run1行）と method_bench_summary.csv（手法ごとの平均・sd・n、素3DGS 30k との差と Welch p）
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

E = Path("/workspace/experiments")
HERE = Path(__file__).parent
EXP = "20260704_wb_r15mm_f80_open"
IT = 30000

# (手法, 初期化, モデルdirのglob)。r1 は既存 run（命名が違うものを含む）
METHODS = [
    ("3dgs", "sfm", ["output_3dgs30k_r*"]),
    ("mip_ks0.1", "sfm", ["output_mip_30k", "output_mip_30k_rerun", "output_ksdec_01_r*"]),
    ("mip_ks0.05", "sfm", ["output_mip_30k_ks005", "output_mip_30k_ks005_rerun", "output_ksdec_005_r*"]),
    ("absgs", "sfm", ["output_absgs_30k_r*"]),
    ("2dgs", "sfm", ["output_2dgs_30k_r*"]),
    ("sadgs", "sfm", ["output_sadgs_30000", "output_sadgs_30000_r*"]),
    ("rain-gs_ours_new", "random", ["output_raings_30k", "output_raings_30k_r*"]),
    ("lodgs", "sfm", ["output_lodgs_30k", "output_lodgs_30k_r*"]),
]


def summary(model, md):
    d = E / EXP / "gt" / (f"wire_eval_{model}_{IT}" + ("_md0" if md == 0 else ""))
    f = d / "summary.json"
    if not f.exists() or json.load(open(f)).get("psnr_wireonly") is None or \
            (md == 2 and "disappearance_by_class" not in json.load(open(f))):
        subprocess.run([sys.executable, str(HERE / "eval_wires.py"), str(E / EXP), str(IT), model,
                        "--mask-dilate", str(md)], check=True, capture_output=True)
    return json.load(open(f))


def r(x, n=2):
    return None if x is None else round(x, n)


rows = []
for method, init, globs in METHODS:
    models = []
    for g in globs:
        models += sorted(p.name for p in (E / EXP).glob(g) if (p / "test" / f"ours_{IT}" / "renders").exists())
    for seed, m in enumerate(dict.fromkeys(models), 1):
        a, b = summary(m, 2), summary(m, 0)
        rows.append({"exp": EXP, "method": method, "init": init, "iters": IT, "run": seed, "model_dir": m,
                     "psnr_wireonly": r(a["psnr_wireonly"]), "psnr_wireonly_md0": r(b["psnr_wireonly"]),
                     "psnr_bar": r(a["psnr_bar"]), "psnr_thin_mixed": r(a["psnr_wire"]),
                     "psnr_bg": r(a["psnr_bg"]), "psnr_all": r(a["psnr_all"]),
                     "gap_wireonly_db": r(a["gap_wireonly_db"]), "psnr_wire_min": r(a["psnr_wire_min"]),
                     "disappearance_norm_wire": r(((a.get("disappearance_by_class") or {}).get("wire") or {}).get("disappearance_norm"), 3),
                     "n_test_views": a["n_test_views"]})
        print(f"{method:18s} {m:28s} wire {rows[-1]['psnr_wireonly']:6.2f}  md0 {rows[-1]['psnr_wireonly_md0']:6.2f}"
              f"  bg {rows[-1]['psnr_bg']:6.2f}")

with open(HERE / "method_bench_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

base = np.array([x["psnr_wireonly"] for x in rows if x["method"] == "3dgs"])
summ = []
for method, init, _ in METHODS:
    v = np.array([x["psnr_wireonly"] for x in rows if x["method"] == method])
    v0 = np.array([x["psnr_wireonly_md0"] for x in rows if x["method"] == method])
    bg = np.array([x["psnr_bg"] for x in rows if x["method"] == method])
    if len(v) == 0:
        continue
    p = stats.ttest_ind(v, base, equal_var=False).pvalue if method != "3dgs" and len(v) > 1 and len(base) > 1 else None
    summ.append({"method": method, "init": init, "iters": IT, "n": len(v),
                 "wire_mean": r(v.mean()), "wire_sd": r(v.std(ddof=1)) if len(v) > 1 else None,
                 "wire_md0_mean": r(v0.mean()), "bg_mean": r(bg.mean()),
                 "d_vs_3dgs": r(v.mean() - base.mean()), "welch_p_vs_3dgs": r(p, 4)})
with open(HERE / "method_bench_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
    w.writeheader()
    w.writerows(summ)
print()
for s in summ:
    print(f"{s['method']:18s} n={s['n']}  wire {s['wire_mean']:6.2f} ± {s['wire_sd']}  md0 {s['wire_md0_mean']:6.2f}"
          f"  bg {s['bg_mean']:6.2f}  Δ3dgs {s['d_vs_3dgs']:+.2f}  p={s['welch_p_vs_3dgs']}")
print(f"saved: method_bench_results.csv（{len(rows)}行）/ method_bench_summary.csv")

# 解像度統制（§9aj）の全runを実験dirの summary.json から集めて resolution_results.csv を作り直す。
# 2026-09-24（監査R-14）：PSNRマスクの膨張（既定2px＝md2）と膨張なし（md0）の両方を列に持つ。
#   res512 では 2px 膨張マスクの約72%が背景画素になり、「1.5pxでは学習延長が無効」は md2 でしか出ない。
#   md0 の summary が無い run は eval_wires.py --mask-dilate 0 をその場で実行する。
# 対象 run は下の RUNS（glob）で決める。追加run（C2/C3 の _r3 等）は命名規則に合えば自動で拾う。
# 使い方: python3 collect_resolution.py   → resolution_results.csv（plot_resolution.py が読む）
import csv
import json
import subprocess
import sys
from pathlib import Path

E = Path("/workspace/experiments")
HERE = Path(__file__).parent

# (実験, 解像度ラベル, 見かけ幅p50, モデル種別, iter, モデルdirのglob)
RUNS = [
    ("20260704_wb_r15mm_f80_open", "1024x768", 3.0, "3dgs", 7000,
     ["output_gen_newdef_r*", "output_gen_noexp_r*", "output_gen_old_r*", "output_gen_oplr05_r*"]),
    ("20260704_wb_r15mm_f80_open", "1024x768", 3.0, "3dgs", 30000, ["output_3dgs30k_r*"]),
    ("20260704_wb_r15mm_f80_open", "1024x768", 3.0, "mip_ks0.1", 30000,
     ["output_mip_30k", "output_mip_30k_rerun", "output_ksdec_01_r*"]),
    ("20260704_wb_r15mm_f80_open", "1024x768", 3.0, "mip_ks0.05", 30000,
     ["output_mip_30k_ks005", "output_mip_30k_ks005_rerun", "output_ksdec_005_r*"]),
    ("20260907_wb_r15mm_f80_open_res512", "512x384", 1.51, "3dgs", 7000, ["output_3dgs7000", "output_3dgs7000_r*"]),
    ("20260907_wb_r15mm_f80_open_res512", "512x384", 1.51, "3dgs", 30000, ["output_3dgs30000", "output_3dgs30000_r*"]),
    ("20260907_wb_r15mm_f80_open_res512", "512x384", 1.51, "mip_ks0.1", 30000, ["output_mip30k_ks01", "output_mip30k_ks01_r*"]),
    ("20260907_wb_r15mm_f80_open_res512", "512x384", 1.51, "mip_ks0.05", 30000, ["output_mip30k_ks005"]),
]


def summary(exp, model, it, md):
    d = E / exp / "gt" / (f"wire_eval_{model}_{it}" + ("_md0" if md == 0 else ""))
    f = d / "summary.json"
    need = not f.exists() or "psnr_wireonly" not in json.load(open(f)) or json.load(open(f))["psnr_wireonly"] is None
    if need:
        subprocess.run([sys.executable, str(HERE / "eval_wires.py"), str(E / exp), str(it), model,
                        "--mask-dilate", str(md)], check=True, capture_output=True)
    return json.load(open(f))


rows = []
for exp, res, px, kind, it, globs in RUNS:
    models = []
    for g in globs:
        models += sorted(p.name for p in (E / exp).glob(g) if (p / "test" / f"ours_{it}" / "renders").exists())
    for seed, m in enumerate(dict.fromkeys(models), 1):
        a, b = summary(exp, m, it, 2), summary(exp, m, it, 0)
        rows.append({"exp": exp, "res": res, "wire_px_p50": px, "model": kind, "iters": it, "seed": seed,
                     "model_dir": m,
                     "psnr_wireonly": round(a["psnr_wireonly"], 2), "psnr_bar": round(a["psnr_bar"], 2),
                     "psnr_thin_mixed": round(a["psnr_wire"], 2), "psnr_bg": round(a["psnr_bg"], 2),
                     "psnr_all": round(a["psnr_all"], 2), "gap_wireonly_db": round(a["gap_wireonly_db"], 2),
                     "disappearance_rate": round(a["disappearance_rate"], 3),
                     "psnr_wireonly_md0": round(b["psnr_wireonly"], 2), "psnr_bar_md0": round(b["psnr_bar"], 2),
                     "gap_wireonly_md0_db": round(b["gap_wireonly_db"], 2)})
        print(f"{exp[-22:]:22s} {kind:10s} {it:6d} {m:28s} md2 {rows[-1]['psnr_wireonly']:6.2f}  md0 {rows[-1]['psnr_wireonly_md0']:6.2f}")

with open(HERE / "resolution_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"saved: {HERE / 'resolution_results.csv'}（{len(rows)}行）")

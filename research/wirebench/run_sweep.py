# 細線ベンチマークの統制変数スイープを一括実行するオーケストレータ
# 各条件（電線太さ×背景×視点数）について 生成→SfM→分析→学習→レンダ→評価 を順に実行し、
# 結果を research/wirebench/sweep_results.csv に1行ずつ追記する。
#
# 使い方:
#   python3 research/wirebench/run_sweep.py \
#       --radii 0.008 0.015 0.03 0.06 --frames 80 --variants open enclosed --iters 7000
#
# 特徴:
#   - 各ステージは出力があればスキップ（中断→再実行で続きから走る）
#   - 学習前にCUDAを確認し、GPUが消えていたら分かるメッセージで停止（exit 2）
#   - 個別実験のログは experiments/<exp>/sweep_*.log に残る

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WS = Path("/workspace")
BLENDER = WS / "tools/blender-4.2.5-linux-x64/blender"
WB = WS / "research/wirebench"
CSV_PATH = WB / "sweep_results.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--radii", type=float, nargs="+", default=[0.008, 0.015, 0.03, 0.06])
parser.add_argument("--frames", type=int, nargs="+", default=[80])
parser.add_argument("--variants", nargs="+", default=["open", "enclosed"],
                    choices=["open", "enclosed"])
parser.add_argument("--iters", type=int, default=7000)
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"),
                    help="実験名の日付プレフィクス（再開時は初回実行日を指定）")
args = parser.parse_args()


def run(cmd, log_path, cwd=None):
    """コマンドを実行しログに書く。失敗したら例外"""
    with open(log_path, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"失敗(exit {r.returncode}): {' '.join(map(str, cmd))}  → {log_path}")


def require_gpu(stage, exp_name):
    import torch
    if not torch.cuda.is_available():
        print(f"\n[中断] GPUが見えません（{exp_name} の {stage} 直前）。"
              f"ホスト側で docker restart 後、同じコマンドで再実行すれば続きから走ります。")
        sys.exit(2)


def process(radius, n_frames, variant):
    name = f"{args.date}_wb_r{int(radius*1000):02d}mm_f{n_frames}_{variant}"
    exp = WS / "experiments" / name
    log = exp / "sweep.log"
    exp.mkdir(parents=True, exist_ok=True)
    print(f"\n━━ {name} ━━", flush=True)

    # 1. シーン生成+レンダリング
    if len(list((exp / "input").glob("*.png"))) >= n_frames:
        print("  1) render: スキップ（生成済み）", flush=True)
    else:
        print("  1) render: 実行", flush=True)
        cmd = [BLENDER, "-b", "-P", WB / "make_scene.py", "--",
               "--out", exp, "--frames", n_frames, "--samples", args.samples,
               "--wire-radius", radius]
        if variant == "enclosed":
            cmd.append("--enclosed")
        run(cmd, log)

    # 2. COLMAP
    if (exp / "sparse/0/points3D.bin").exists():
        print("  2) colmap: スキップ", flush=True)
    else:
        print("  2) colmap: 実行", flush=True)
        require_gpu("colmap", name)
        run([sys.executable, WS / "scripts/run_colmap.py",
             "--source_path", exp, "--camera_model", "PINHOLE"], log)

    # 3. SfM分析（常に実行・軽い）
    print("  3) analyze_sfm: 実行", flush=True)
    run([sys.executable, WB / "analyze_sfm.py", "."], log, cwd=exp)

    # 4. 3DGS学習
    if (exp / f"output/point_cloud/iteration_{args.iters}").exists():
        print("  4) train: スキップ", flush=True)
    else:
        print("  4) train: 実行", flush=True)
        require_gpu("train", name)
        run([sys.executable, WS / "scripts/run_train.py",
             "--source", exp, "--model_path", exp / "output",
             "--iterations", args.iters, "--save_iterations", args.iters,
             "--test_iterations", args.iters, "--eval"], log)

    # 5. テスト視点レンダ
    if list((exp / f"output/test/ours_{args.iters}/renders").glob("*.png")):
        print("  5) render_test: スキップ", flush=True)
    else:
        print("  5) render_test: 実行", flush=True)
        require_gpu("render", name)
        run([sys.executable, WS / "scripts/run_render.py",
             "-m", exp / "output", "-s", exp,
             "--iteration", args.iters, "--skip_train"], log)

    # 6. 細線評価
    print("  6) eval_wires: 実行", flush=True)
    run([sys.executable, WB / "eval_wires.py", ".", str(args.iters)], log, cwd=exp)

    # 7. 集計行をCSVへ
    sfm = json.load(open(exp / "gt/sfm_analysis.json"))
    ev = json.load(open(exp / f"gt/wire_eval_{args.iters}/summary.json"))
    sp = json.load(open(exp / "gt/scene_params.json"))
    row = {
        "exp": name, "wire_radius_m": radius, "frames": n_frames, "variant": variant,
        "iters": args.iters,
        "sfm_points": sfm["n_points3d"], "wire_span_pts_3cm": sfm["wire_span_points"]["3cm"],
        "thin_px_share_pct": round(sfm["thin_pixel_share_pct"], 3),
        "psnr_all": round(ev["psnr_all"], 2), "psnr_wire": round(ev["psnr_wire"], 2),
        "psnr_bg": round(ev["psnr_bg"], 2), "gap_db": round(ev["gap_db"], 2),
        "psnr_wire_min": round(ev["psnr_wire_min"], 2),
        "align_rmse_m": round(sfm["align_rmse_m"], 4), "samples": sp["samples"],
    }
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        # 同名実験の既存行は許容（最新行を採用する運用）
        w.writerow(row)
    print(f"  ✔ 完了: gap={row['gap_db']}dB wire={row['psnr_wire']}dB "
          f"(SfM点 {row['wire_span_pts_3cm']})", flush=True)


combos = [(r, f, v) for v in args.variants for f in args.frames for r in args.radii]
print(f"スイープ開始: {len(combos)}条件  (radii={args.radii} frames={args.frames} variants={args.variants})")
for i, (r, f, v) in enumerate(combos, 1):
    print(f"\n[{i}/{len(combos)}]", flush=True)
    process(r, f, v)
print(f"\n全条件完了。結果: {CSV_PATH}")

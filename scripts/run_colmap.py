# COLMAP を使ってカメラ姿勢推定を実行するラッパースクリプト
# 4ステップ（特徴点抽出→マッチング→マッパー→undistortion）を個別実行し
# "[COLMAP N/4]" 形式の進捗マーカーを標準出力に出力する

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list) -> int:
    print(f"実行: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="COLMAPでカメラ姿勢推定を行う")
    parser.add_argument("--source_path", "-s", required=True,
                        help="実験ディレクトリのパス（input/ フォルダを含む）")
    parser.add_argument("--camera_model", default="OPENCV",
                        help="カメラモデル（OPENCV / PINHOLE / SIMPLE_RADIAL）")
    parser.add_argument("--no_gpu", action="store_true", help="GPUを使わない")
    args = parser.parse_args()

    source    = Path(args.source_path)
    input_dir = source / "input"

    if not input_dir.exists():
        print(f"ERROR: input/ フォルダが見つかりません: {input_dir}", file=sys.stderr)
        print("フレーム抽出を先に実行してください。", file=sys.stderr)
        sys.exit(1)

    n_images = len(list(input_dir.glob("*.jpg"))) + len(list(input_dir.glob("*.png")))
    print(f"入力画像数: {n_images} 枚", flush=True)

    use_gpu          = "0" if args.no_gpu else "1"
    db_path          = source / "distorted" / "database.db"
    sparse_distorted = source / "distorted" / "sparse"
    sparse_distorted.mkdir(parents=True, exist_ok=True)

    # ── [1/4] 特徴点抽出 ──────────────────────────────────────────────────────
    print("[COLMAP 1/4] 特徴点抽出を開始...", flush=True)
    ret = run_cmd([
        "colmap", "feature_extractor",
        "--database_path",           str(db_path),
        "--image_path",              str(input_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", args.camera_model,
        "--SiftExtraction.use_gpu",  use_gpu,
    ])
    if ret != 0:
        print(f"ERROR: 特徴点抽出が失敗しました (code {ret})", file=sys.stderr)
        sys.exit(ret)

    # ── [2/4] 特徴点マッチング ────────────────────────────────────────────────
    print("[COLMAP 2/4] 特徴点マッチングを開始...", flush=True)
    ret = run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path",       str(db_path),
        "--SiftMatching.use_gpu", use_gpu,
    ])
    if ret != 0:
        print(f"ERROR: 特徴点マッチングが失敗しました (code {ret})", file=sys.stderr)
        sys.exit(ret)

    # ── [3/4] マッパー（3D再構成） ────────────────────────────────────────────
    print("[COLMAP 3/4] マッパー（3D再構成）を開始...", flush=True)
    ret = run_cmd([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(input_dir),
        "--output_path",   str(sparse_distorted),
        "--Mapper.ba_global_function_tolerance=0.000001",
    ])
    if ret != 0:
        print(f"ERROR: マッパーが失敗しました (code {ret})", file=sys.stderr)
        sys.exit(ret)

    # ── [4/4] 画像undistortion ────────────────────────────────────────────────
    print("[COLMAP 4/4] 画像undistortionを開始...", flush=True)
    ret = run_cmd([
        "colmap", "image_undistorter",
        "--image_path",  str(input_dir),
        "--input_path",  str(sparse_distorted / "0"),
        "--output_path", str(source),
        "--output_type", "COLMAP",
    ])
    if ret != 0:
        print(f"ERROR: undistortionが失敗しました (code {ret})", file=sys.stderr)
        sys.exit(ret)

    # ── sparse フォルダ整理 ───────────────────────────────────────────────────
    # image_undistorter は sparse/ に直接モデルファイルを出力するため sparse/0/ に移動
    sparse_dir = source / "sparse"
    sparse_0   = sparse_dir / "0"
    sparse_0.mkdir(parents=True, exist_ok=True)
    for f in sparse_dir.iterdir():
        if f.is_file():
            shutil.move(str(f), str(sparse_0 / f.name))

    print(f"完了: {source}", flush=True)


if __name__ == "__main__":
    main()

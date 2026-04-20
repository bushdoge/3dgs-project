# COLMAP（gaussian-splatting/convert.py経由）を使ってカメラ姿勢推定を実行するラッパースクリプト
# 実験ディレクトリ直下の input/ フォルダを入力として受け取り、sparse/0/ と images/ を生成する

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="COLMAPでカメラ姿勢推定を行う")
    parser.add_argument("--source_path", "-s", required=True,
                        help="実験ディレクトリのパス（input/ フォルダを含む）")
    parser.add_argument("--camera_model", default="OPENCV",
                        help="カメラモデル（OPENCV / PINHOLE / SIMPLE_RADIAL）")
    parser.add_argument("--no_gpu", action="store_true", help="GPUを使わない")
    args = parser.parse_args()

    source = Path(args.source_path)
    input_dir = source / "input"

    if not input_dir.exists():
        print(f"ERROR: input/ フォルダが見つかりません: {input_dir}", file=sys.stderr)
        print("フレーム抽出を先に実行してください。", file=sys.stderr)
        sys.exit(1)

    n_images = len(list(input_dir.glob("*.jpg"))) + len(list(input_dir.glob("*.png")))
    print(f"入力画像数: {n_images} 枚", flush=True)

    cmd = [
        "python", "/opt/gaussian-splatting/convert.py",
        "--source_path", str(source),
        "--camera", args.camera_model,
    ]
    if args.no_gpu:
        cmd.append("--no_gpu")

    print(f"実行コマンド: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

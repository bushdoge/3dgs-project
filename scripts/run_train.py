# 3D Gaussian Splattingの学習（train.py）を実行するラッパースクリプト
# 実験ディレクトリ（sparse/0/ と images/ を含む）を入力として学習を実行する

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="3DGS学習を実行する")
    parser.add_argument("--source", "-s", required=True,
                        help="実験ディレクトリのパス（sparse/0/ を含む）")
    parser.add_argument("--model_path", required=True,
                        help="モデル出力先フォルダ")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="学習ステップ数（デフォルト: 30000）")
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7000, 30000], help="チェックポイント保存タイミング")
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7000, 30000], help="評価（PSNR計算）タイミング")
    args = parser.parse_args()

    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "/opt/gaussian-splatting/train.py",
        "-s", args.source,
        "--model_path", args.model_path,
        "--iterations", str(args.iterations),
        "--save_iterations", *[str(i) for i in args.save_iterations],
        "--test_iterations", *[str(i) for i in args.test_iterations],
    ]

    print(f"実行コマンド: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

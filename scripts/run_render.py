# 3D Gaussian Splatting の render.py を実行するラッパースクリプト
# 学習済みモデルから全カメラ視点の画像をレンダリングする

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="3DGSレンダリングを実行する")
    parser.add_argument("--model_path", "-m", required=True,
                        help="学習済みモデルのフォルダ（output/）")
    parser.add_argument("--source_path", "-s", required=True,
                        help="実験ディレクトリ（input/・sparse/ を含む）")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="レンダリングするイテレーション（-1 で最新を自動選択）")
    parser.add_argument("--skip_train", action="store_true",
                        help="学習データ視点のレンダリングをスキップ")
    parser.add_argument("--skip_test", action="store_true",
                        help="テストデータ視点のレンダリングをスキップ")
    parser.add_argument("--white_background", action="store_true",
                        help="背景を白にする（デフォルト: 黒）")
    args = parser.parse_args()

    cmd = [
        sys.executable, "/opt/gaussian-splatting/render.py",
        "-m", args.model_path,
        "-s", args.source_path,
        "--iteration", str(args.iteration),
        "--quiet",
    ]
    if args.skip_train:        cmd.append("--skip_train")
    if args.skip_test:         cmd.append("--skip_test")
    if args.white_background:  cmd.append("--white_background")

    print(f"[RUN_RENDER] コマンド: {' '.join(cmd)}", flush=True)
    print(f"[RUN_RENDER] モデル  : {args.model_path}", flush=True)
    print(f"[RUN_RENDER] ソース  : {args.source_path}", flush=True)
    print(f"[RUN_RENDER] iter   : {args.iteration if args.iteration >= 0 else '最新'}", flush=True)

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print("[RUN_RENDER] レンダリング完了！", flush=True)
    else:
        print(f"[RUN_RENDER] ERROR: 終了コード {result.returncode}", flush=True)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

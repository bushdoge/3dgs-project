# 動画ファイルからFFmpegを使って連番画像（フレーム）を切り出すスクリプト

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="動画から連番フレームを抽出する")
    parser.add_argument("--input", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output", required=True, help="出力フォルダのパス（experiment/input/ を推奨）")
    parser.add_argument("--fps", type=float, default=2.0, help="抽出するFPS（デフォルト: 2.0）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", args.input,
        "-vf", f"fps={args.fps}",
        "-q:v", "2",
        str(output_dir / "frame_%06d.jpg"),
        "-y",
    ]

    print(f"実行コマンド: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, text=True, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    n_frames = len(list(output_dir.glob("*.jpg")))
    print(f"抽出完了: {n_frames} 枚 → {output_dir}", flush=True)


if __name__ == "__main__":
    main()

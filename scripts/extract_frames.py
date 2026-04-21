# 動画ファイルからFFmpegを使って連番画像（フレーム）を切り出すスクリプト
# 進捗は "PROGRESS_TOTAL N" と "PROGRESS cur/total" 形式で標準出力に出力する

import argparse
import subprocess
import sys
import time
from pathlib import Path


def get_estimated_frames(input_path: str, fps: float) -> int:
    """ffprobeで動画の長さを取得し抽出フレーム数を推定する"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", input_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
        return max(1, int(duration * fps))
    except (ValueError, TypeError):
        return 0


def main():
    parser = argparse.ArgumentParser(description="動画から連番フレームを抽出する")
    parser.add_argument("--input",  required=True, help="入力動画ファイルのパス")
    parser.add_argument("--output", required=True, help="出力フォルダのパス（experiment/input/ を推奨）")
    parser.add_argument("--fps", type=float, default=2.0, help="抽出するFPS（デフォルト: 2.0）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = get_estimated_frames(args.input, args.fps)
    if total > 0:
        print(f"PROGRESS_TOTAL {total}", flush=True)
        print(f"総フレーム数（推定）: {total} 枚", flush=True)

    cmd = [
        "ffmpeg", "-i", args.input,
        "-vf", f"fps={args.fps}",
        "-q:v", "2",
        str(output_dir / "frame_%06d.jpg"),
        "-y",
    ]
    print(f"実行コマンド: {' '.join(cmd)}", flush=True)

    # ffmpegを起動し、出力ファイル数をポーリングして進捗を出力する
    proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        count = len(list(output_dir.glob("*.jpg")))
        print(f"PROGRESS {count}/{total or '?'}", flush=True)
        time.sleep(1)
    proc.wait()

    if proc.returncode != 0:
        print("ERROR: ffmpegが失敗しました", file=sys.stderr)
        sys.exit(1)

    n_frames = len(list(output_dir.glob("*.jpg")))
    print(f"PROGRESS {n_frames}/{n_frames}", flush=True)
    print(f"抽出完了: {n_frames} 枚 → {output_dir}", flush=True)


if __name__ == "__main__":
    main()

# 360度（等距円筒：Equirectangular）動画・画像をピンホールカメラ視点の画像群に変換するスクリプト
# 1枚の360度画像から前・左・右・後・上・下の6方向または指定方向にクロップして出力する

import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def equirect_to_perspective(img, fov_deg, yaw_deg, pitch_deg, out_w, out_h):
    """等距円筒画像から指定方向のピンホール画像を切り出す。

    fov_deg  : 水平視野角（度）
    yaw_deg  : 水平回転（度、右が正）
    pitch_deg: 垂直回転（度、上が正）
    """
    h, w = img.shape[:2]
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    f = (out_w / 2) / math.tan(fov / 2)

    # 出力ピクセルごとに球面座標を計算
    u = np.linspace(-(out_w - 1) / 2, (out_w - 1) / 2, out_w)
    v = np.linspace(-(out_h - 1) / 2, (out_h - 1) / 2, out_h)
    uu, vv = np.meshgrid(u, v)

    # カメラ座標 → 球面座標
    x = uu
    y = vv
    z = np.full_like(x, f)

    # pitch回転（X軸）
    cp, sp = math.cos(pitch), math.sin(pitch)
    x2 = x
    y2 = y * cp - z * sp
    z2 = y * sp + z * cp

    # yaw回転（Y軸）
    cy, sy = math.cos(yaw), math.sin(yaw)
    x3 = x2 * cy + z2 * sy
    y3 = y2
    z3 = -x2 * sy + z2 * cy

    r = np.sqrt(x3**2 + y3**2 + z3**2)
    lon = np.arctan2(x3, z3)
    lat = np.arcsin(np.clip(y3 / r, -1, 1))

    # 等距円筒のピクセル座標へ変換
    map_x = ((lon / (2 * math.pi)) + 0.5) * w
    map_y = (0.5 - lat / math.pi) * h
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


DIRECTIONS = {
    "front":  (0,   0),
    "right":  (90,  0),
    "back":   (180, 0),
    "left":   (270, 0),
    "up":     (0,  90),
    "down":   (0, -90),
}


def process_image(img_path, out_dir, fov, out_w, out_h, directions):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  読み込みエラー: {img_path}", flush=True)
        return 0

    stem = img_path.stem
    count = 0
    for name in directions:
        yaw, pitch = DIRECTIONS[name]
        out = equirect_to_perspective(img, fov, yaw, pitch, out_w, out_h)
        out_path = out_dir / f"{stem}_{name}.jpg"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    return count


def extract_frames(video_path, frames_dir, fps):
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(frames_dir / "frame_%06d.jpg"),
        "-y",
    ]
    result = subprocess.run(cmd, text=True, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return sorted(frames_dir.glob("*.jpg"))


def main():
    parser = argparse.ArgumentParser(description="360度動画・画像をピンホール画像群に変換する")
    parser.add_argument("--input", required=True,
                        help="入力ファイル（動画 or 画像）またはフォルダ（画像群）")
    parser.add_argument("--output", required=True, help="出力フォルダ（変換後画像の保存先）")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="水平視野角（度）デフォルト: 90")
    parser.add_argument("--width", type=int, default=1024, help="出力画像の幅（デフォルト: 1024）")
    parser.add_argument("--height", type=int, default=1024, help="出力画像の高さ（デフォルト: 1024）")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="動画入力時のフレーム抽出FPS（デフォルト: 1.0）")
    parser.add_argument("--directions", nargs="+",
                        default=["front", "right", "back", "left"],
                        choices=list(DIRECTIONS.keys()),
                        help="変換する方向（デフォルト: front right back left）")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 入力が動画の場合はフレーム抽出してから変換
    if input_path.is_file() and input_path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
        print(f"動画からフレームを抽出中（{args.fps} fps）...", flush=True)
        tmp_frames = Path("/workspace/tmp") / f"360frames_{input_path.stem}"
        images = extract_frames(input_path, tmp_frames, args.fps)
        print(f"  → {len(images)} フレーム抽出", flush=True)
    elif input_path.is_dir():
        images = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    elif input_path.is_file():
        images = [input_path]
    else:
        print(f"ERROR: 入力が見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for i, img_path in enumerate(images, 1):
        n = process_image(img_path, out_dir, args.fov, args.width, args.height, args.directions)
        total += n
        if i % 10 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] 変換中...", flush=True)

    print(f"変換完了: {total} 枚 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()

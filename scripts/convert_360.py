# 360度（等距円筒：Equirectangular）動画・画像をピンホールカメラ視点の画像群に変換するスクリプト
# 1枚の360度画像から前・左・右・後・上・下の6方向または指定方向にクロップして出力する

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def build_equirect_maps(h, w, fov_deg, yaw_deg, pitch_deg, out_w, out_h):
    """等距円筒（h×w）→ピンホール視点のremapテーブル（map_x, map_y）を作る。
    同じ視点のフレームを大量に変換するときはこれを1回作って使い回す
    （generate_masks.py のマスク投影でも同じテーブルを使い、画像とマスクの
    ピクセル対応を厳密に保つ）。

    fov_deg  : 水平視野角（度）
    yaw_deg  : 水平回転（度、右が正）
    pitch_deg: 垂直回転（度、上が正）
    """
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    f = (out_w / 2) / math.tan(fov / 2)

    # 出力ピクセルごとに球面座標を計算
    # vv: 上端が負・下端が正（画像座標）
    # y = -vv とすることで、上端が正（空方向）・下端が負（地面方向）に修正
    u = np.linspace(-(out_w - 1) / 2, (out_w - 1) / 2, out_w)
    v = np.linspace(-(out_h - 1) / 2, (out_h - 1) / 2, out_h)
    uu, vv = np.meshgrid(u, v)

    # カメラ座標 → 球面座標（Y上向き系）
    x = uu
    y = -vv   # 上端=正latitude（空）、下端=負latitude（地面）に修正
    z = np.full_like(x, f)

    # pitch回転（X軸、正=上向き）
    # y2 = y*cp + z*sp, z2 = -y*sp + z*cp で正pitchが上方向になる
    cp, sp = math.cos(pitch), math.sin(pitch)
    x2 = x
    y2 = y * cp + z * sp
    z2 = -y * sp + z * cp

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
    return map_x, map_y


def equirect_to_perspective(img, fov_deg, yaw_deg, pitch_deg, out_w, out_h):
    """等距円筒画像から指定方向のピンホール画像を切り出す"""
    h, w = img.shape[:2]
    map_x, map_y = build_equirect_maps(h, w, fov_deg, yaw_deg, pitch_deg, out_w, out_h)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def process_image(img_path, out_dir, fov, out_w, out_h, angles,
                  eq_dir=None, eq_width=2048):
    """angles: list of (yaw_deg, pitch_deg) tuples
    eq_dir を指定すると、変換元の等距円筒フレームも保存する
    （SAM2マスク生成用。eq_width に縮小してストレージを節約する）"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  読み込みエラー: {img_path}", flush=True)
        return 0

    stem = img_path.stem
    if eq_dir is not None:
        eq = img
        if eq_width and img.shape[1] > eq_width:
            eq_h = round(img.shape[0] * eq_width / img.shape[1])
            eq = cv2.resize(img, (eq_width, eq_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(eq_dir / f"{stem}.jpg"), eq, [cv2.IMWRITE_JPEG_QUALITY, 95])

    count = 0
    for yaw, pitch in angles:
        out = equirect_to_perspective(img, fov, yaw, pitch, out_w, out_h)
        out_path = out_dir / f"{stem}_y{int(yaw):03d}_p{int(pitch):+d}.jpg"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    return count


def extract_frames(video_path, frames_dir, fps):
    # 前回実行の残骸フレームが混入しないように、一時フォルダを空にしてから抽出する
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    # -y は出力ファイルより前に置く必要がある（後ろだと trailing option として無視される）
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(frames_dir / "frame_%06d.jpg"),
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
    parser.add_argument("--angles", nargs="+",
                        default=["0,0", "90,0", "180,0", "270,0"],
                        help="変換する角度ペア yaw,pitch（例: 0,0 45,30 90,-30）")
    parser.add_argument("--keep-equirect", default=None, metavar="DIR",
                        help="等距円筒フレームの保存先フォルダ（SAM2マスク生成用）。"
                             "変換パラメータも DIR/meta.json に記録する")
    parser.add_argument("--equirect-width", type=int, default=2048,
                        help="--keep-equirect 保存時の幅（デフォルト: 2048。0で原寸）")
    args = parser.parse_args()

    angles = []
    for a in args.angles:
        yaw_s, pitch_s = a.split(",")
        angles.append((float(yaw_s), float(pitch_s)))

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_dir = None
    if args.keep_equirect:
        eq_dir = Path(args.keep_equirect)
        # 前回実行の残骸が混入しないように既存フレームを削除する
        if eq_dir.exists():
            for f in eq_dir.glob("*.jpg"):
                f.unlink()
        eq_dir.mkdir(parents=True, exist_ok=True)

    # 入力が動画の場合はフレーム抽出してから変換
    tmp_frames = None
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
        n = process_image(img_path, out_dir, args.fov, args.width, args.height, angles,
                          eq_dir=eq_dir, eq_width=args.equirect_width)
        total += n
        print(f"  [{i}/{len(images)}] 変換中...", flush=True)

    # 一時フレームを削除してストレージを節約する
    if tmp_frames is not None:
        shutil.rmtree(tmp_frames, ignore_errors=True)

    if eq_dir is not None:
        # マスク投影（generate_masks.py --equirect）が同じremapを再現するための変換パラメータ
        meta = {"fov": args.fov, "width": args.width, "height": args.height,
                "angles": angles}
        (eq_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"等距円筒フレーム保存: {len(images)} 枚 → {eq_dir}", flush=True)

    print(f"変換完了: {total} 枚 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()

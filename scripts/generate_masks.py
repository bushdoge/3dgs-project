# SAM2を使って実験のinput/フォルダの画像から撮影者マスクを生成するスクリプト
# 360度動画由来の画像は方向別（y=000, y=045 など）に分けてSAM2を実行する
# クリック座標をJSON指定 → 各方向の時系列フレームに伝播 → masks/フォルダに保存

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

SAM2_CHECKPOINT = "/workspace/models/pretrained/sam2/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ── フレームグループ化 ──────────────────────────────────────────────────────────

def group_frames_by_direction(input_dir: Path) -> dict[str, list[Path]]:
    """
    ファイル名から方向（y000, y045 等）を抽出してグループ化する。
    360度変換画像（_y\d+_p 形式）でない場合は 'all' として全フレームを1グループにまとめる。
    """
    exts = {".jpg", ".jpeg", ".png"}
    all_frames = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in exts])

    groups: dict[str, list[Path]] = {}
    pattern = re.compile(r'_y(\d+)_p')

    for f in all_frames:
        m = pattern.search(f.stem)
        if m:
            key = f"y{m.group(1)}"
        else:
            key = "all"
        groups.setdefault(key, []).append(f)

    # 各グループ内はファイル名順（時系列順）でソート済み
    return groups


def print_groups(groups: dict[str, list[Path]]):
    print(f"\n方向グループ: {len(groups)} グループ")
    for key, frames in sorted(groups.items()):
        print(f"  {key}: {len(frames)} フレーム  （先頭: {frames[0].name}）")


# ── SAM2 ───────────────────────────────────────────────────────────────────────

def load_predictor():
    from sam2.build_sam import build_sam2_video_predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    return predictor, device


def run_sam2_on_group(predictor, frames: list[Path], clicks: list[tuple],
                       masks_dir: Path, direction: str):
    """1方向分のフレーム列にSAM2を実行してマスクを保存する"""
    masks_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_map = {}
        for i, f in enumerate(frames):
            dst = tmpdir / f"{i:05d}.jpg"
            img = cv2.imread(str(f))
            cv2.imwrite(str(dst), img)
            frame_map[i] = f

        print(f"  [{direction}] {len(frames)} フレームを推論中...", flush=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(tmpdir))
            predictor.reset_state(state)

            pts    = np.array([[x, y] for x, y, _ in clicks], dtype=np.float32)
            labels = np.array([l for _, _, l in clicks],       dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=pts,
                labels=labels,
            )

            saved = 0
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                orig_path = frame_map[frame_idx]
                mask_path = masks_dir / (orig_path.stem + ".png")
                mask = (masks[0][0].cpu().numpy() > 0.0).astype(np.uint8) * 255
                cv2.imwrite(str(mask_path), mask)
                saved += 1

    print(f"  [{direction}] 完了: {saved} 枚保存", flush=True)


# ── SOR ────────────────────────────────────────────────────────────────────────

def apply_sor(exp_dir: Path, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """points3D.txt に SOR を適用して points3D_clean.txt を出力する"""
    import open3d as o3d

    pts_path = exp_dir / "sparse" / "0" / "points3D.txt"
    out_path = exp_dir / "sparse" / "0" / "points3D_clean.txt"

    if not pts_path.exists():
        print("[SOR] points3D.txt が見つかりません")
        return

    points, colors, lines_meta = [], [], []
    with open(pts_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            points.append([float(p[1]), float(p[2]), float(p[3])])
            colors.append([int(p[4]), int(p[5]), int(p[6])])
            lines_meta.append(line)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)

    print(f"[SOR] 入力: {len(points):,} 点", flush=True)
    _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_set = set(inlier_idx)
    print(f"[SOR] 残存: {len(inlier_idx):,} 点  除去: {len(points)-len(inlier_idx):,} 点", flush=True)

    with open(out_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, line in enumerate(lines_meta):
            if i in inlier_set:
                f.write(line)

    print(f"[SOR] 保存: {out_path}", flush=True)


# ── メイン ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM2で方向別にマスク生成 + SORで点群クリーニング",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("exp_dir",
        help="実験ディレクトリ（input/フォルダを含む）")
    parser.add_argument("--clicks-json", default=None,
        help="クリック座標 [[x,y,label],...] をJSON文字列で指定\n"
             "  label=1: 撮影者（マスクする）  label=0: 背景\n"
             "  例: --clicks-json '[[512,900,1]]'")
    parser.add_argument("--directions", default=None,
        help="処理する方向をカンマ区切りで指定（省略時は全方向）\n"
             "  例: --directions y000,y090,y180,y270")
    parser.add_argument("--list-directions", action="store_true",
        help="方向グループの一覧を表示して終了")
    parser.add_argument("--sor-only",      action="store_true",
        help="SAM2をスキップしてSORのみ実行")
    parser.add_argument("--sam-only",      action="store_true",
        help="SORをスキップしてSAM2のみ実行")
    parser.add_argument("--sor-neighbors", type=int,   default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    args = parser.parse_args()

    exp_dir   = Path(args.exp_dir)
    input_dir = exp_dir / "input"
    masks_dir = exp_dir / "masks"

    if not input_dir.exists():
        print(f"ERROR: input/ フォルダが見つかりません: {input_dir}")
        sys.exit(1)

    groups = group_frames_by_direction(input_dir)
    if not groups:
        print("ERROR: input/ に画像が見つかりません")
        sys.exit(1)

    # 方向一覧の表示のみ
    if args.list_directions:
        print_groups(groups)
        return

    print_groups(groups)

    # ── SAM2 ──────────────────────────────────────────────────────────────────
    if not args.sor_only:
        if not args.clicks_json:
            print("\nERROR: --clicks-json でクリック座標を指定してください。")
            print("  例: --clicks-json '[[512, 900, 1]]'")
            print("\n座標の確認方法:")
            print("  python3 -c \"")
            print(f"  from PIL import Image")
            print(f"  img = Image.open('{input_dir}/{sorted(groups.values())[0][0].name}')")
            print(f"  print(img.size)  # (width, height)\"")
            sys.exit(1)

        raw    = json.loads(args.clicks_json)
        clicks = [(x, y, l) for x, y, l in raw]
        print(f"\nクリック座標: {clicks}")

        # 対象方向を絞り込む
        target_dirs = set(args.directions.split(",")) if args.directions else set(groups.keys())
        to_process  = {k: v for k, v in sorted(groups.items()) if k in target_dirs}

        if not to_process:
            print(f"ERROR: 指定した方向が見つかりません: {args.directions}")
            print(f"  利用可能: {', '.join(sorted(groups.keys()))}")
            sys.exit(1)

        print(f"\n[SAM2] {len(to_process)} 方向を処理します")
        predictor, device = load_predictor()
        print(f"[SAM2] デバイス: {device}\n")

        for direction, frames in to_process.items():
            run_sam2_on_group(predictor, frames, clicks, masks_dir, direction)

        total = sum(len(v) for v in to_process.values())
        print(f"\n[SAM2] 全処理完了: {total} 枚 → {masks_dir}")

    # ── SOR ───────────────────────────────────────────────────────────────────
    if not args.sam_only:
        try:
            apply_sor(exp_dir, args.sor_neighbors, args.sor_std_ratio)
        except ImportError:
            print("[SOR] open3d が未インストールのためスキップ（pip install open3d）")
        except Exception as e:
            print(f"[SOR] エラー: {e}")


if __name__ == "__main__":
    main()

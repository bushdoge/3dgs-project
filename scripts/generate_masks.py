# SAM2を使って実験のinput/フォルダの画像から撮影者マスクを生成するスクリプト
# 360度動画由来の画像は方向別（y=000, y=045 など）に分けてSAM2を実行する
# クリック座標をJSON指定 → 各方向の時系列フレームに伝播 → masks/フォルダに保存
# クリック座標は全方向共通のリスト [[x,y,label],...] または
# 方向別の辞書 {"y000": [[x,y,label],...], ...} のどちらでも指定できる
# 方向別の値は {"frame": N, "points": [[x,y,label],...]} 形式も可（フレームNにプロンプトを
# 与え、そこから前後両方向に伝播する。1枚目に撮影者が写っていないシーン用）
# （torch は重いので必要になるまで import しない。GUIページからの部分importを軽くするため）

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

SAM2_CHECKPOINT = "/workspace/models/pretrained/sam2/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ── フレームグループ化 ──────────────────────────────────────────────────────────

def group_frames_by_direction(input_dir: Path) -> dict[str, list[Path]]:
    """
    ファイル名から方向（y000_p+0, y045_p+30 等）を抽出してグループ化する。
    同じyawでもピッチが違えば別視点の動画なので、yawとpitchの組でグループ化する
    （yawだけで分けると y090_p+0 と y090_p+30 が混ざり、1枚ごとに視点が飛ぶ
    シーケンスがSAM2に渡ってしまう）。
    360度変換画像（_y\d+_p 形式）でない場合は 'all' として全フレームを1グループにまとめる。
    """
    exts = {".jpg", ".jpeg", ".png"}
    all_frames = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in exts])

    groups: dict[str, list[Path]] = {}
    pattern = re.compile(r'_y(\d+)_p([+-]?\d+)')

    for f in all_frames:
        m = pattern.search(f.stem)
        if m:
            key = f"y{m.group(1)}_p{m.group(2)}"
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
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告: CUDAが利用できません。CPUで実行します（非常に時間がかかります）", flush=True)
    predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    return predictor, device


def run_sam2_on_group(predictor, device: str, frames: list[Path], clicks: list[tuple],
                       masks_dir: Path, direction: str, ann_frame_idx: int = 0):
    """1方向分のフレーム列にSAM2を実行してマスクを保存する。
    ann_frame_idx のフレームにクリック点を与え、そこから順方向（→末尾）と
    逆方向（→先頭）の2回伝播で全フレームをカバーする"""
    import contextlib
    import torch

    masks_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_map = {}
        for i, f in enumerate(frames):
            dst = tmpdir / f"{i:05d}.jpg"
            img = cv2.imread(str(f))
            cv2.imwrite(str(dst), img)
            frame_map[i] = f

        ann_frame_idx = max(0, min(ann_frame_idx, len(frames) - 1))
        print(f"  [{direction}] {len(frames)} フレームを推論中..."
              f"（プロンプト: フレーム {ann_frame_idx} = {frames[ann_frame_idx].name}）", flush=True)

        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else contextlib.nullcontext())
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(tmpdir))
            predictor.reset_state(state)

            pts    = np.array([[x, y] for x, y, _ in clicks], dtype=np.float32)
            labels = np.array([l for _, _, l in clicks],       dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=ann_frame_idx,
                obj_id=1,
                points=pts,
                labels=labels,
            )

            saved_idx = set()
            total = len(frames)
            passes = [False] if ann_frame_idx == 0 else [False, True]
            for reverse in passes:
                for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, reverse=reverse):
                    if frame_idx in saved_idx:   # プロンプトフレームは両方向で重複する
                        continue
                    saved_idx.add(frame_idx)
                    orig_path = frame_map[frame_idx]
                    mask_path = masks_dir / (orig_path.stem + ".png")
                    mask = (masks[0][0].cpu().numpy() > 0.0).astype(np.uint8) * 255
                    cv2.imwrite(str(mask_path), mask)
                    if len(saved_idx) % 10 == 0 or len(saved_idx) == total:
                        print(f"PROGRESS {direction} {len(saved_idx)}/{total}", flush=True)
            saved = len(saved_idx)

    print(f"  [{direction}] 完了: {saved} 枚保存", flush=True)


# ── SOR ────────────────────────────────────────────────────────────────────────

def _sor_filter_model(model_dir: Path, nb_neighbors: int, std_ratio: float):
    """COLMAPモデル（points3D.bin）にSORを適用してその場で書き戻す。
    各点のk近傍平均距離が「全体平均 + std_ratio×標準偏差」を超える点を外れ値として除去する
    （open3d の remove_statistical_outlier と同じアルゴリズム）。
    元のモデルは before_sor/ にバックアップする。"""
    import shutil
    import pycolmap
    from scipy.spatial import cKDTree

    rec = pycolmap.Reconstruction(str(model_dir))
    n = rec.num_points3D()
    if n <= nb_neighbors + 1:
        print(f"[SOR] {model_dir}: 点数が少なすぎるためスキップ（{n} 点）", flush=True)
        return

    ids = np.array(list(rec.points3D.keys()))
    xyz = np.array([rec.points3D[i].xyz for i in ids])

    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=nb_neighbors + 1)   # 先頭は自分自身（距離0）
    mean_d = dists[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std()
    outlier_ids = ids[mean_d > thresh]

    print(f"[SOR] {model_dir}", flush=True)
    print(f"[SOR]   入力: {n:,} 点 → 残存: {n - len(outlier_ids):,} 点  除去: {len(outlier_ids):,} 点", flush=True)

    if len(outlier_ids) == 0:
        return

    # バックアップ（初回のみ）。サブフォルダはCOLMAPリーダーに無視されるので安全
    backup_dir = model_dir / "before_sor"
    if not backup_dir.exists():
        backup_dir.mkdir()
        for f in ("cameras.bin", "images.bin", "points3D.bin"):
            if (model_dir / f).exists():
                shutil.copy2(model_dir / f, backup_dir / f)
        print(f"[SOR]   バックアップ: {backup_dir}", flush=True)

    for pid in outlier_ids:
        rec.delete_point3D(int(pid))
    rec.write(str(model_dir))

    # gaussian-splattingは初回にbin→plyへ変換した points3D.ply をキャッシュとして使うため、
    # 古いplyが残っているとSOR結果が反映されない。削除して再生成させる
    stale_ply = model_dir / "points3D.ply"
    if stale_ply.exists():
        stale_ply.unlink()
        print(f"[SOR]   古い points3D.ply を削除（次回学習時に再生成されます）", flush=True)


def apply_sor(exp_dir: Path, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """実験内のCOLMAPモデルにSORを適用する。
    3DGS学習が実際に点群を読むのは undistortion 後の dense/sparse/0/ なので、
    sparse/0/ と dense/sparse/0/ の両方（存在するもの）に適用する。"""
    targets = [
        exp_dir / "sparse" / "0",
        exp_dir / "dense" / "sparse" / "0",
    ]
    found = False
    for model_dir in targets:
        if (model_dir / "points3D.bin").exists():
            found = True
            _sor_filter_model(model_dir, nb_neighbors, std_ratio)
    if not found:
        print("[SOR] points3D.bin が見つかりません（先に姿勢推定を実行してください）")


# ── メイン ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM2で方向別にマスク生成 + SORで点群クリーニング",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("exp_dir",
        help="実験ディレクトリ（input/フォルダを含む）")
    parser.add_argument("--clicks-json", default=None,
        help="クリック座標をJSON文字列で指定\n"
             "  label=1: 撮影者（マスクする）  label=0: 背景\n"
             "  全方向共通: --clicks-json '[[512,900,1]]'\n"
             "  方向別:     --clicks-json '{\"y000\": [[512,900,1]], \"y090\": [[300,800,1]]}'\n"
             "  プロンプトフレーム指定（1枚目に撮影者が写っていない場合）:\n"
             "    --clicks-json '{\"y000\": {\"frame\": 30, \"points\": [[512,900,1]]}}'\n"
             "    → フレーム30にクリック点を与え、前後両方向に伝播する")
    parser.add_argument("--directions", default=None,
        help="処理する方向をカンマ区切りで指定（省略時は全方向）\n"
             "  例: --directions y000_p+0,y090_p+0\n"
             "  方向名は --list-directions で確認できます")
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

        raw = json.loads(args.clicks_json)

        def _parse_entry(v) -> tuple[int, list[tuple]]:
            """[[x,y,l],...] / {"frame": N, "points": [[x,y,l],...]} を (フレームidx, クリック列) に揃える"""
            if isinstance(v, dict):
                return int(v.get("frame", 0)), [(x, y, l) for x, y, l in v.get("points", [])]
            return 0, [(x, y, l) for x, y, l in v]

        if isinstance(raw, dict):
            # 方向別のクリック座標 {"y000": [[x,y,l],...] または {"frame":N,"points":[...]}, ...}
            clicks_by_dir = {}
            for k, v in raw.items():
                frame_idx, pts = _parse_entry(v)
                if pts:
                    clicks_by_dir[k] = (frame_idx, pts)
        else:
            # 全方向共通のクリック座標 [[x,y,l],...]
            common = _parse_entry(raw)
            clicks_by_dir = {k: common for k in groups.keys()}
        print("\nクリック座標:")
        for k, (frame_idx, pts) in sorted(clicks_by_dir.items()):
            print(f"  {k}: フレーム {frame_idx} に {pts}")

        # 対象方向を絞り込む（--directions指定 > クリック辞書のキー > 全方向）
        if args.directions:
            target_dirs = set(args.directions.split(","))
        else:
            target_dirs = set(clicks_by_dir.keys())
        to_process = {k: v for k, v in sorted(groups.items()) if k in target_dirs}

        if not to_process:
            print(f"ERROR: 指定した方向が見つかりません: {args.directions or list(clicks_by_dir)}")
            print(f"  利用可能: {', '.join(sorted(groups.keys()))}")
            sys.exit(1)

        missing = [d for d in to_process if d not in clicks_by_dir]
        if missing:
            print(f"警告: クリック座標が未指定の方向をスキップします: {', '.join(missing)}")
            to_process = {k: v for k, v in to_process.items() if k in clicks_by_dir}

        print(f"\n[SAM2] {len(to_process)} 方向を処理します")
        predictor, device = load_predictor()
        print(f"[SAM2] デバイス: {device}\n")

        for direction, frames in to_process.items():
            ann_frame_idx, dir_clicks = clicks_by_dir[direction]
            run_sam2_on_group(predictor, device, frames, dir_clicks,
                              masks_dir, direction, ann_frame_idx)

        total = sum(len(v) for v in to_process.values())
        print(f"\n[SAM2] 全処理完了: {total} 枚 → {masks_dir}")

    # ── SOR ───────────────────────────────────────────────────────────────────
    if not args.sam_only:
        try:
            apply_sor(exp_dir, args.sor_neighbors, args.sor_std_ratio)
        except ImportError as e:
            print(f"[SOR] 依存ライブラリ不足のためスキップ: {e}")
        except Exception as e:
            print(f"[SOR] エラー: {e}")


if __name__ == "__main__":
    main()

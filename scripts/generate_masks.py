# SAM2を使って実験のinput/フォルダの画像から撮影者マスクを生成するスクリプト
# 360度動画由来の画像は方向別（y=000, y=045 など）に分けてSAM2を実行する
# クリック座標をJSON指定 → 各方向の時系列フレームに伝播 → masks/フォルダに保存
# クリック座標は全方向共通のリスト [[x,y,label],...] または
# 方向別の辞書 {"y000": [[x,y,label],...], ...} のどちらでも指定できる
# 方向別の値は {"frame": N, "points": [[x,y,label],...]} 形式も可（フレームNにプロンプトを
# 与え、そこから前後両方向に伝播する。1枚目に撮影者が写っていないシーン用）
# --equirect: equirect/ の等距円筒フレームでSAM2を1系統実行し、出来たマスクを
# meta.json の変換パラメータ（convert_360.pyと同じremap）で全ピンホール視点に投影する
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


def sam2_propagate(predictor, device: str, frames: list[Path], ann_frame_idx: int = 0,
                   roll_x: int = 0, clicks: list[tuple] | None = None,
                   mask_prompt: np.ndarray | None = None,
                   progress_tag: str | None = None) -> dict[int, np.ndarray]:
    """フレーム列にSAM2を伝播させ {フレームidx: 世界座標マスク(uint8, 0/255)} を返す。
    プロンプトは clicks（点）か mask_prompt（2値マスク・世界座標）のどちらかで与える。
    roll_x: 横方向にroll_xピクセル回してから推論し、結果は逆rollで世界座標に戻す
    （等距円筒の左右端の継ぎ目で対象が分断されるのを防ぐ）。"""
    import contextlib
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_w = None
        for i, f in enumerate(frames):
            img = cv2.imread(str(f))
            if img_w is None:
                img_w = img.shape[1]
            if roll_x:
                img = np.roll(img, roll_x, axis=1)
            cv2.imwrite(str(tmpdir / f"{i:05d}.jpg"), img)

        ann_frame_idx = max(0, min(ann_frame_idx, len(frames) - 1))

        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else contextlib.nullcontext())
        out: dict[int, np.ndarray] = {}
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(tmpdir))
            predictor.reset_state(state)

            if mask_prompt is not None:
                mp = np.roll(mask_prompt, roll_x, axis=1) if roll_x else mask_prompt
                predictor.add_new_mask(
                    inference_state=state, frame_idx=ann_frame_idx,
                    obj_id=1, mask=(mp > 127),
                )
            else:
                pts    = np.array([[(x + roll_x) % img_w, y] for x, y, _ in clicks],
                                  dtype=np.float32)
                labels = np.array([l for _, _, l in clicks], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=ann_frame_idx,
                    obj_id=1, points=pts, labels=labels,
                )

            total = len(frames)
            passes = [False] if ann_frame_idx == 0 else [False, True]
            for reverse in passes:
                for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, reverse=reverse):
                    if frame_idx in out:   # プロンプトフレームは両方向で重複する
                        continue
                    mask = (masks[0][0].cpu().numpy() > 0.0).astype(np.uint8) * 255
                    if roll_x:
                        mask = np.roll(mask, -roll_x, axis=1)
                    out[frame_idx] = mask
                    if progress_tag and (len(out) % 10 == 0 or len(out) == total):
                        print(f"PROGRESS {progress_tag} {len(out)}/{total}", flush=True)
    return out


def run_sam2_on_group(predictor, device: str, frames: list[Path], clicks: list[tuple],
                       masks_dir: Path, direction: str, ann_frame_idx: int = 0,
                       roll_x: int = 0):
    """1方向分のフレーム列にSAM2を実行してマスクをmasks_dirに保存する（方向別モード）。"""
    masks_dir.mkdir(parents=True, exist_ok=True)
    ann = max(0, min(ann_frame_idx, len(frames) - 1))
    print(f"  [{direction}] {len(frames)} フレームを推論中..."
          f"（プロンプト: フレーム {ann} = {frames[ann].name}）", flush=True)
    out = sam2_propagate(predictor, device, frames, ann, roll_x,
                         clicks=clicks, progress_tag=direction)
    for idx, mask in out.items():
        cv2.imwrite(str(masks_dir / (frames[idx].stem + ".png")), mask)
    print(f"  [{direction}] 完了: {len(out)} 枚保存", flush=True)


# ── 等距円筒モード ──────────────────────────────────────────────────────────────

def load_equirect_frames(exp_dir: Path) -> tuple[list[Path], dict]:
    """equirect/ のフレーム一覧と meta.json（変換パラメータ）を読み込む"""
    eq_dir = exp_dir / "equirect"
    meta_path = eq_dir / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} が見つかりません。")
        print("  --equirect は convert_360.py --keep-equirect で抽出した実験のみ対応です。")
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted([f for f in eq_dir.iterdir() if f.suffix.lower() in exts])
    if not frames:
        print(f"ERROR: {eq_dir} に画像が見つかりません")
        sys.exit(1)
    return frames, meta


def compute_roll_x(clicks: list[tuple], eq_width: int) -> int:
    """最初の撮影者クリック点（label=1）が画像中央に来る横roll量を返す。
    等距円筒の左右端の継ぎ目で対象が分断されるのを防ぐ"""
    for x, _, l in clicks:
        if l == 1:
            return (eq_width // 2 - int(x)) % eq_width
    return 0


def _circular_centroid_x(mask: np.ndarray, eq_width: int):
    """マスク（uint8）の横方向の重心xを円環として求める（左右端をまたぐ対象に対応）。
    マスクが空なら None を返す。"""
    cols = (mask > 127).sum(axis=0).astype(np.float64)
    if cols.sum() == 0:
        return None
    ang = 2 * np.pi * np.arange(eq_width) / eq_width
    cx = (cols * np.cos(ang)).sum()
    sx = (cols * np.sin(ang)).sum()
    if cx == 0 and sx == 0:
        return None
    return (np.arctan2(sx, cx) % (2 * np.pi)) * eq_width / (2 * np.pi)


def _pick_pass_b_prompt(masks_a: dict[int, np.ndarray], click_x: int, eq_width: int):
    """パスB用のプロンプトフレームとマスクを選ぶ。
    パスAの継ぎ目（click+180°）から最も離れた = パスAが最も信頼できる位置のうち、
    パスBの中央寄り（click±90°付近）に撮影者がいるフレームを選ぶ。
    そのフレームのパスAマスクをそのままBのmask_promptに使う。
    戻り値: (フレームidx, マスク) / 見つからなければ (None, None)"""
    best = None  # (スコア, idx)
    for target in ((click_x + eq_width // 4) % eq_width,
                   (click_x - eq_width // 4) % eq_width):
        for idx, m in masks_a.items():
            cxx = _circular_centroid_x(m, eq_width)
            if cxx is None:
                continue
            # 円環上での目標位置からの距離（近いほど良い）。マスク面積も加味する
            d = abs((cxx - target + eq_width / 2) % eq_width - eq_width / 2)
            area = (m > 127).sum()
            if area < 50:               # 小さすぎるマスクは信頼しない
                continue
            score = d - 0.0  # 距離最小を優先
            if best is None or score < best[0]:
                best = (score, idx)
    if best is None:
        return None, None
    return best[1], masks_a[best[1]]


def run_sam2_equirect_union(predictor, device, frames, clicks, eq_masks_dir,
                            ann_frame_idx, eq_width):
    """等距円筒フレームに2パスでSAM2を実行し、マスクをOR合成してeq_masks_dirに保存する。
    パスA: 撮影者を中央化（継ぎ目は180°反対側）。
    パスB: 継ぎ目を180°ずらし、パスAの良好なマスクをseedに追跡。
    撮影者がカメラ周りを一周移動しても、各パスが担当する半周ずつをunionで埋められる。"""
    eq_masks_dir.mkdir(parents=True, exist_ok=True)
    click_x = next((int(x) for x, _, l in clicks if l == 1), eq_width // 2)

    # ── パスA（撮影者を中央化）──
    roll_a = compute_roll_x(clicks, eq_width)
    print(f"  [パスA] 撮影者を中央化（roll={roll_a}px）", flush=True)
    masks_a = sam2_propagate(predictor, device, frames, ann_frame_idx, roll_a,
                             clicks=clicks, progress_tag="passA")

    # ── パスB（継ぎ目を180°ずらす）──
    roll_b = (roll_a + eq_width // 2) % eq_width
    fb, prompt_mask = _pick_pass_b_prompt(masks_a, click_x, eq_width)
    if fb is not None:
        print(f"  [パスB] 継ぎ目を180°ずらす（roll={roll_b}px、"
              f"プロンプト: フレーム {fb} のパスAマスク）", flush=True)
        masks_b = sam2_propagate(predictor, device, frames, fb, roll_b,
                                 mask_prompt=prompt_mask, progress_tag="passB")
    else:
        # パスAが空だったフォールバック：元クリックでそのままB座標で追跡
        print(f"  [パスB] 継ぎ目を180°ずらす（roll={roll_b}px、プロンプト: 元クリック）", flush=True)
        masks_b = sam2_propagate(predictor, device, frames, ann_frame_idx, roll_b,
                                 clicks=clicks, progress_tag="passB")

    # ── OR合成 ──
    saved = 0
    for i, f in enumerate(frames):
        a = masks_a.get(i)
        b = masks_b.get(i)
        if a is None and b is None:
            continue
        if a is None:
            u = b
        elif b is None:
            u = a
        else:
            u = np.maximum(a, b)
        cv2.imwrite(str(eq_masks_dir / (f.stem + ".png")), u)
        saved += 1
    print(f"  [union] 完了: {saved} 枚保存", flush=True)
    return saved


def project_masks_to_pinhole(input_dir: Path, masks_dir: Path, eq_masks_dir: Path,
                             meta: dict, dilate_px: int = 7) -> int:
    """等距円筒マスクを meta の変換パラメータで各ピンホール視点に投影して
    masks/ に保存する。画像変換（convert_360.py）と同じremapテーブルを使うため
    ピンホール画像とマスクのピクセル対応は厳密に一致する。
    マスクは2値なので最近傍補間とし、境界の取りこぼしを防ぐため投影前に
    dilate_px ピクセル膨張させて安全側（消しすぎ側）に倒す。
    input/ に対応するピンホール画像が存在するフレームのみ保存する"""
    from convert_360 import build_equirect_maps

    mask_files = sorted(eq_masks_dir.glob("*.png"))
    if not mask_files:
        print("ERROR: 等距円筒マスクが見つかりません")
        sys.exit(1)

    sample = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    eq_h, eq_w = sample.shape
    out_w, out_h = int(meta["width"]), int(meta["height"])

    # 方向ごとのremapテーブルは全フレーム共通なので1回だけ作る
    maps = []
    for yaw, pitch in meta["angles"]:
        map_x, map_y = build_equirect_maps(eq_h, eq_w, meta["fov"], yaw, pitch, out_w, out_h)
        suffix = f"_y{int(yaw):03d}_p{int(pitch):+d}"
        maps.append((suffix, map_x, map_y))

    kernel = np.ones((dilate_px, dilate_px), np.uint8) if dilate_px > 0 else None
    input_exts = (".jpg", ".jpeg", ".png")
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [投影] {len(mask_files)} フレーム × {len(maps)} 方向...", flush=True)
    saved = 0
    total = len(mask_files)
    for i, mf in enumerate(mask_files, 1):
        m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if kernel is not None:
            m = cv2.dilate(m, kernel)
        for suffix, map_x, map_y in maps:
            stem = mf.stem + suffix
            if not any((input_dir / f"{stem}{e}").exists() for e in input_exts):
                continue
            pm = cv2.remap(m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
            cv2.imwrite(str(masks_dir / f"{stem}.png"), pm)
            saved += 1
        if i % 10 == 0 or i == total:
            print(f"PROGRESS project {i}/{total}", flush=True)

    print(f"  [投影] 完了: {saved} 枚保存", flush=True)
    return saved


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

def parse_clicks_entry(v) -> tuple[int, list[tuple]]:
    """[[x,y,l],...] / {"frame": N, "points": [[x,y,l],...]} を (フレームidx, クリック列) に揃える"""
    if isinstance(v, dict):
        return int(v.get("frame", 0)), [(x, y, l) for x, y, l in v.get("points", [])]
    return 0, [(x, y, l) for x, y, l in v]


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
    parser.add_argument("--equirect", action="store_true",
        help="等距円筒モード：equirect/ のフレームでSAM2を1系統実行し、\n"
             "マスクを全ピンホール視点に投影する（--keep-equirect付きで抽出した実験用）。\n"
             "クリック座標は等距円筒画像上で指定: --clicks-json '{\"equirect\": {\"frame\": 0, \"points\": [[x,y,1]]}}'")
    parser.add_argument("--mask-dilate", type=int, default=7,
        help="等距円筒モードの投影前にマスクを膨張させるpx数（デフォルト: 7。0で無効）")
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

    # ── SAM2（等距円筒モード）─────────────────────────────────────────────────
    if args.equirect and not args.sor_only:
        eq_frames, meta = load_equirect_frames(exp_dir)
        if not args.clicks_json:
            print("\nERROR: --clicks-json でクリック座標を指定してください（等距円筒画像上の座標）。")
            print("  例: --clicks-json '{\"equirect\": {\"frame\": 0, \"points\": [[1024, 900, 1]]}}'")
            sys.exit(1)

        raw = json.loads(args.clicks_json)
        entry = raw.get("equirect", raw) if isinstance(raw, dict) else raw
        ann_frame_idx, clicks = parse_clicks_entry(entry)
        if not clicks:
            print("ERROR: クリック座標が空です")
            sys.exit(1)

        eq_w = cv2.imread(str(eq_frames[0])).shape[1]
        print(f"\n[SAM2/等距円筒] {len(eq_frames)} フレーム　プロンプト: フレーム {ann_frame_idx}　"
              f"（2パスunion：撮影者が一周移動しても継ぎ目で途切れないようにする）")

        predictor, device = load_predictor()
        print(f"[SAM2] デバイス: {device}\n")

        eq_masks_dir = exp_dir / "masks_equirect"
        run_sam2_equirect_union(predictor, device, eq_frames, clicks,
                                eq_masks_dir, ann_frame_idx, eq_w)
        saved = project_masks_to_pinhole(input_dir, masks_dir, eq_masks_dir, meta,
                                         args.mask_dilate)
        print(f"\n[SAM2] 全処理完了: {saved} 枚 → {masks_dir}")

    # ── SAM2（方向別モード）───────────────────────────────────────────────────
    elif not args.sor_only:
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
        if isinstance(raw, dict):
            # 方向別のクリック座標 {"y000": [[x,y,l],...] または {"frame":N,"points":[...]}, ...}
            clicks_by_dir = {}
            for k, v in raw.items():
                frame_idx, pts = parse_clicks_entry(v)
                if pts:
                    clicks_by_dir[k] = (frame_idx, pts)
        else:
            # 全方向共通のクリック座標 [[x,y,l],...]
            common = parse_clicks_entry(raw)
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

# 3D Gaussian Splattingの学習（train.py）を実行するラッパースクリプト
# 実験ディレクトリ（sparse/0/ と input/ または images/ を含む）を入力として学習を実行する
# カメラモデルが PINHOLE/SIMPLE_PINHOLE でない場合は自動で undistortion を実行してから学習する

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PINHOLE_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


def calc_auto_resolution(source: Path) -> int | None:
    """VRAMと画像枚数・サイズから必要な縮小倍率を自動計算する。縮小不要なら None を返す"""
    try:
        import math
        import torch
        from PIL import Image

        if not torch.cuda.is_available():
            return None

        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        # 画像ロードはVRAMの40%以内に収める（残りを学習用ガウシアンに確保）
        budget_bytes = vram_bytes * 0.40

        images_dir = None
        for candidate in ["images", "input"]:
            p = source / candidate
            if p.exists():
                images_dir = p
                break
        if images_dir is None:
            return None

        imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        n = len(imgs)
        if n == 0:
            return None

        w, h = Image.open(imgs[0]).size
        total_bytes = n * w * h * 3 * 4  # float32 RGB

        label = f"{n}枚 × {w}×{h}px = {total_bytes/1e9:.1f}GB / 予算 {budget_bytes/1e9:.1f}GB"
        if total_bytes <= budget_bytes:
            print(f"[解像度自動判定] {label} → 縮小不要"
                  "（ただし幅1600px超の画像はgaussian-splatting標準動作で1.6Kに縮小されます。"
                  "フル解像度にしたい場合は --resolution 1 を明示してください）", flush=True)
            return None

        r = math.sqrt(total_bytes / budget_bytes)
        for res in [2, 4, 8]:
            if res >= r:
                print(f"[解像度自動判定] {label} → --resolution {res} を自動設定", flush=True)
                return res

        print(f"[解像度自動判定] {label} → --resolution 8 を自動設定", flush=True)
        return 8

    except Exception as e:
        print(f"[解像度自動判定] 計算失敗（{e}）→ 縮小なし", flush=True)
        return None


def get_camera_models(sparse_0: Path) -> set:
    """sparse/0/ のカメラモデル一覧を返す（必要なら .bin → .txt 変換）"""
    cameras_bin = sparse_0 / "cameras.bin"
    cameras_txt = sparse_0 / "cameras.txt"

    if cameras_bin.exists() and not cameras_txt.exists():
        subprocess.run(
            ["colmap", "model_converter",
             "--input_path", str(sparse_0),
             "--output_path", str(sparse_0),
             "--output_type", "TXT"],
            check=True, capture_output=True,
        )

    if not cameras_txt.exists():
        return set()

    models = set()
    for line in cameras_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            models.add(parts[1])
    return models


def build_mask_undistorter(orig_sparse0: Path, undist_sparse0: Path):
    """歪みありモデルとundistort後モデルから、マスクを再マップする関数を返す。
    マスクは歪みあり画像（input/）上で生成されるため、undistort後の画像に
    そのまま重ねると位置がずれる。undistort後画像の各ピクセルを歪みモデルで
    元画像座標へ射影し、マスクを再サンプリングして整合させる。
    モデルが読めない場合は None を返す（呼び出し側はリサイズのみにフォールバック）。"""
    try:
        import cv2
        import numpy as np
        import pycolmap
        rec_d = pycolmap.Reconstruction(str(orig_sparse0))
        rec_u = pycolmap.Reconstruction(str(undist_sparse0))
    except Exception as e:
        print(f"[masks] カメラモデル読み込み失敗: {e}", flush=True)
        return None

    # 画像名 → (歪みありカメラID, undistort後カメラID)
    names_u = {im.name: im.camera_id for im in rec_u.images.values()}
    name_to_cams = {im.name: (im.camera_id, names_u[im.name])
                    for im in rec_d.images.values() if im.name in names_u}
    maps_cache = {}

    def undistort(mask, image_name: str):
        key = name_to_cams.get(image_name)
        if key is None:
            return None
        if key not in maps_cache:
            cam_d = rec_d.cameras[key[0]]
            cam_u = rec_u.cameras[key[1]]
            K = cam_u.calibration_matrix()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            u, v = np.meshgrid(np.arange(cam_u.width), np.arange(cam_u.height))
            pts = np.stack([(u.ravel() - cx) / fx,
                            (v.ravel() - cy) / fy,
                            np.ones(u.size)], axis=1)
            uv = cam_d.img_from_cam(pts)   # 歪みモデルを適用して元画像ピクセルへ
            maps_cache[key] = (
                uv[:, 0].reshape(cam_u.height, cam_u.width).astype(np.float32),
                uv[:, 1].reshape(cam_u.height, cam_u.width).astype(np.float32),
                int(cam_d.width), int(cam_d.height),
            )
        mx, my, dw, dh = maps_cache[key]
        if (mask.shape[1], mask.shape[0]) != (dw, dh):
            mask = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        return cv2.remap(mask, mx, my, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return undistort


def apply_masks_to_images(source: Path, masks_dir: Path, orig_dir: Path = None) -> str:
    """
    マスク画像（masks/）を学習画像にアルファチャンネルとして合成し
    images_masked/ に RGBA PNG として保存する。
    マスク白(255)=撮影者領域=学習除外、黒(0)=通常領域。
    orig_dir: マスク生成元の実験ディレクトリ。source と異なる（=undistortion済み）
              場合はマスクをカメラモデルに合わせて再マップする。
    返り値: gaussian-splatting に渡す --images サブフォルダ名
    """
    import cv2
    import numpy as np
    from PIL import Image

    # 学習画像ディレクトリを特定
    img_dir = None
    for candidate in ["images", "input"]:
        p = source / candidate
        if p.exists():
            img_dir = p
            break
    if img_dir is None:
        print("[masks] 画像フォルダが見つかりません。マスクをスキップします。", flush=True)
        return "images"

    out_dir = source / "images_masked"
    out_dir.mkdir(exist_ok=True)

    # undistortionが行われている場合はマスクも同じカメラモデルで再マップする
    undistorter = None
    if orig_dir is not None and source.resolve() != Path(orig_dir).resolve():
        undistorter = build_mask_undistorter(
            Path(orig_dir) / "sparse" / "0", source / "sparse" / "0"
        )
        if undistorter is not None:
            print("[masks] undistortion検出 → マスクもカメラモデルに合わせて再マップします", flush=True)
        else:
            print("[masks] 警告: カメラモデルを読めないため、マスクはリサイズのみで適用します", flush=True)

    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    applied = 0
    for img_path in imgs:
        mask_path = masks_dir / (img_path.stem + ".png")
        # COLMAPのimages.binには元の拡張子込みのファイル名が登録されており、
        # gaussian-splattingはその名前のままimagesフォルダから画像を開く。
        # そのためファイル名は変えず、中身だけRGBA対応のPNG形式で保存する
        # （PILは拡張子ではなくファイル内容で形式を判別するため読み込み可能）。
        out_path  = out_dir / img_path.name

        img = np.array(Image.open(img_path).convert("RGB"))

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if undistorter is not None:
                    remapped = undistorter(mask, img_path.name)
                    if remapped is not None:
                        mask = remapped
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
                # 白=撮影者 → アルファ0（除外）、黒=背景 → アルファ255（学習）
                alpha = (255 - mask).astype(np.uint8)
                applied += 1
            else:
                alpha = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        else:
            alpha = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)

        rgba = np.dstack([img, alpha])
        Image.fromarray(rgba, "RGBA").save(str(out_path), format="PNG")

    print(f"[masks] {applied}/{len(imgs)} 枚にマスクを適用 → {out_dir}", flush=True)
    return "images_masked"


def run_undistortion(source: Path) -> Path:
    """undistortion を実行し、学習用ソースパス（dense/）を返す"""
    dense = source / "dense"

    if dense.exists():
        print(f"[undistortion] 既存の dense/ を使用: {dense}", flush=True)
        # sparse/0/ が未整理の場合は整理する
        dense_sparse   = dense / "sparse"
        dense_sparse_0 = dense_sparse / "0"
        if dense_sparse.exists() and not dense_sparse_0.exists():
            dense_sparse_0.mkdir(parents=True, exist_ok=True)
            for f in dense_sparse.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(dense_sparse_0 / f.name))
        return dense

    # 入力画像ディレクトリを探す（input/ → images/ の順で探す）
    input_dir = None
    for candidate in ["input", "images"]:
        p = source / candidate
        if p.exists():
            input_dir = p
            break

    if input_dir is None:
        print("ERROR: input/ または images/ フォルダが見つかりません", file=sys.stderr)
        sys.exit(1)

    print(f"[undistortion] PINHOLE以外のカメラモデルを検出。自動でundistortionを実行します...", flush=True)
    ret = subprocess.run([
        "colmap", "image_undistorter",
        "--image_path",  str(input_dir),
        "--input_path",  str(source / "sparse" / "0"),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
    ]).returncode

    if ret != 0:
        print(f"ERROR: undistortionが失敗しました (code {ret})", file=sys.stderr)
        sys.exit(ret)

    # image_undistorter は sparse/ 直下にファイルを置くので sparse/0/ に整理する
    dense_sparse   = dense / "sparse"
    dense_sparse_0 = dense_sparse / "0"
    if dense_sparse.exists() and not dense_sparse_0.exists():
        dense_sparse_0.mkdir(parents=True, exist_ok=True)
        for f in dense_sparse.iterdir():
            if f.is_file():
                shutil.move(str(f), str(dense_sparse_0 / f.name))

    print(f"[undistortion] 完了: {dense}", flush=True)
    return dense


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
                        default=[1000, 3000, 7000, 15000, 30000], help="評価（PSNR計算）タイミング")
    parser.add_argument("--eval", action="store_true",
                        help="train/test分割を有効化（8枚に1枚をtestに割り当て）")
    parser.add_argument("--resolution", "-r", type=int, default=None,
                        help="画像縮小倍率（例: 2=1/2, 4=1/4）。省略時はVRAMから自動判定")
    args = parser.parse_args()

    source = Path(args.source)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    # カメラモデルチェック → 必要なら自動undistortion
    sparse_0 = source / "sparse" / "0"
    if sparse_0.exists():
        camera_models = get_camera_models(sparse_0)
        if camera_models and not camera_models.issubset(PINHOLE_MODELS):
            print(f"[カメラモデル] {camera_models} → undistortionが必要です", flush=True)
            source = run_undistortion(source)
        else:
            print(f"[カメラモデル] {camera_models} → undistortion不要", flush=True)

    # マスクが存在する場合は images_masked/ を作成して使用
    orig_exp_dir = Path(args.source)
    masks_dir = orig_exp_dir / "masks"
    images_subdir = "images"
    if masks_dir.exists() and any(masks_dir.iterdir()):
        print(f"[masks] masks/ フォルダを検出。マスク合成を実行します...", flush=True)
        images_subdir = apply_masks_to_images(source, masks_dir, orig_exp_dir)
    else:
        # undistortion後のsourceにも images があるので既定値のまま
        for candidate in ["images", "input"]:
            if (source / candidate).exists():
                images_subdir = candidate
                break

    cmd = [
        sys.executable, "/workspace/scripts/train_custom.py",
        "-s", str(source),
        "--model_path", args.model_path,
        "--iterations", str(args.iterations),
        "--save_iterations", *[str(i) for i in args.save_iterations],
        "--test_iterations", *[str(i) for i in args.test_iterations],
        "--images", images_subdir,
    ]
    if args.eval:
        cmd.append("--eval")

    # resolution: 明示指定がなければVRAMから自動判定
    resolution = args.resolution if args.resolution is not None else calc_auto_resolution(source)
    if resolution is not None:
        cmd += ["--resolution", str(resolution)]

    print(f"実行コマンド: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

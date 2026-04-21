# HLoc（SuperPoint/DISK/SIFT + LightGlue/SuperGlue等）を使ってカメラ姿勢推定を行うスクリプト
# 出力は COLMAP 互換形式（sparse/0/）で gaussian-splatting にそのまま渡せる

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/opt/hloc")

from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction


def main():
    parser = argparse.ArgumentParser(description="HLocでカメラ姿勢推定を行う")
    parser.add_argument("--source_path", "-s", required=True,
                        help="実験ディレクトリのパス（input/ フォルダを含む）")
    parser.add_argument("--feature_type", default="superpoint_max",
                        choices=["superpoint_max", "superpoint_aachen", "disk",
                                 "aliked-n16", "sift", "r2d2", "d2net-ss"],
                        help="特徴点抽出器（デフォルト: superpoint_max）")
    parser.add_argument("--matcher_type", default="superpoint+lightglue",
                        choices=["superpoint+lightglue", "disk+lightglue",
                                 "aliked+lightglue", "superglue", "superglue-fast",
                                 "NN-superpoint", "NN-ratio", "NN-mutual"],
                        help="特徴点マッチャー（デフォルト: superpoint+lightglue）")
    args = parser.parse_args()

    source = Path(args.source_path)
    images = source / "input"
    outputs = source / "hloc_outputs"
    sfm_dir = source / "sparse" / "0"

    if not images.exists():
        print(f"ERROR: input/ フォルダが見つかりません: {images}", file=sys.stderr)
        sys.exit(1)

    n_images = len(list(images.glob("*.jpg"))) + len(list(images.glob("*.png")))
    print(f"入力画像数: {n_images} 枚", flush=True)

    outputs.mkdir(parents=True, exist_ok=True)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    features_path = outputs / "features.h5"
    sfm_pairs = outputs / "pairs-sfm.txt"
    matches_path = outputs / "matches.h5"

    # 特徴点抽出
    print(f"[1/4] 特徴点抽出: {args.feature_type}", flush=True)
    feature_conf = extract_features.confs[args.feature_type]
    extract_features.main(feature_conf, images, feature_path=features_path)

    # ペアリスト生成（全ペアの組み合わせ）
    print("[2/4] ペアリスト生成（exhaustive）...", flush=True)
    pairs_from_exhaustive.main(sfm_pairs, image_list=None, features=features_path)

    # 特徴点マッチング
    print(f"[3/4] 特徴点マッチング: {args.matcher_type}", flush=True)
    matcher_conf = match_features.confs[args.matcher_type]
    match_features.main(matcher_conf, sfm_pairs,
                        features=features_path, matches=matches_path)

    # SfM再構成
    print("[4/4] SfM再構成...", flush=True)
    model = reconstruction.main(
        sfm_dir, images, sfm_pairs, features_path, matches_path
    )

    if model is None:
        print("ERROR: 再構成に失敗しました。", file=sys.stderr)
        sys.exit(1)

    print(f"完了: {len(model.images)} カメラ, {len(model.points3D)} 点群", flush=True)
    print(f"出力先: {sfm_dir}", flush=True)


if __name__ == "__main__":
    main()

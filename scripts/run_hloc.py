# HLoc（SuperPoint/DISK/SIFT + LightGlue/SuperGlue等）を使ってカメラ姿勢推定を行うスクリプト
# 出力は COLMAP 互換形式（sparse/0/）で gaussian-splatting にそのまま渡せる
# ペアリスト生成方式: exhaustive（全ペア・4ステップ）または retrieval（類似画像のみ・5ステップ）
# 最後に colmap image_undistorter を実行し、PINHOLE モデルの undistorted/ を生成する

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/hloc")

from hloc import (
    extract_features, match_features,
    pairs_from_exhaustive, pairs_from_retrieval,
    reconstruction,
)


def main():
    parser = argparse.ArgumentParser(description="HLocでカメラ姿勢推定を行う")
    parser.add_argument("--source_path", "-s", required=True,
                        help="実験ディレクトリのパス（input/ フォルダを含む）")
    parser.add_argument("--feature_type", default="superpoint_max",
                        choices=["superpoint_max", "superpoint_aachen", "disk",
                                 "aliked-n16", "sift", "r2d2", "d2net-ss"],
                        help="局所特徴点抽出器（デフォルト: superpoint_max）")
    parser.add_argument("--matcher_type", default="superpoint+lightglue",
                        choices=["superpoint+lightglue", "disk+lightglue",
                                 "aliked+lightglue", "superglue", "superglue-fast",
                                 "NN-superpoint", "NN-ratio", "NN-mutual"],
                        help="特徴点マッチャー（デフォルト: superpoint+lightglue）")
    parser.add_argument("--pair_method", default="exhaustive",
                        choices=["exhaustive", "retrieval"],
                        help="ペアリスト生成方式（デフォルト: exhaustive）")
    parser.add_argument("--retrieval_model", default="netvlad",
                        choices=["netvlad", "openibl", "dir", "megaloc"],
                        help="retrieval方式で使うグローバル特徴量モデル（デフォルト: netvlad）")
    parser.add_argument("--num_matched", type=int, default=20,
                        help="retrieval方式で1画像あたりマッチングするペア数（デフォルト: 20）")
    args = parser.parse_args()

    source  = Path(args.source_path)
    images  = source / "input"
    outputs = source / "hloc_outputs"
    sfm_dir = source / "sparse" / "0"

    if not images.exists():
        print(f"ERROR: input/ フォルダが見つかりません: {images}", file=sys.stderr)
        sys.exit(1)

    n_images = len(list(images.glob("*.jpg"))) + len(list(images.glob("*.png")))
    print(f"入力画像数: {n_images} 枚", flush=True)
    print(f"ペアリスト生成方式: {args.pair_method}", flush=True)

    outputs.mkdir(parents=True, exist_ok=True)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    features_path = outputs / "features.h5"
    sfm_pairs     = outputs / "pairs-sfm.txt"
    matches_path  = outputs / "matches.h5"

    use_retrieval = (args.pair_method == "retrieval")
    total         = 5 if use_retrieval else 4

    # ── [1/N] 局所特徴点抽出 ──────────────────────────────────────────────────
    print(f"[1/{total}] 局所特徴点抽出: {args.feature_type}", flush=True)
    feature_conf = extract_features.confs[args.feature_type]
    extract_features.main(feature_conf, images, feature_path=features_path)

    if use_retrieval:
        # ── [2/5] グローバル特徴量抽出 ──────────────────────────────────────
        print(f"[2/{total}] グローバル特徴量抽出: {args.retrieval_model}", flush=True)
        retrieval_conf   = extract_features.confs[args.retrieval_model]
        global_feat_path = outputs / f"global-feats-{args.retrieval_model}.h5"
        extract_features.main(retrieval_conf, images, feature_path=global_feat_path)

        # ── [3/5] ペアリスト生成（retrieval top-K） ──────────────────────────
        print(f"[3/{total}] ペアリスト生成（retrieval top-{args.num_matched}）...", flush=True)
        pairs_from_retrieval.main(
            global_feat_path, sfm_pairs, num_matched=args.num_matched
        )
    else:
        # ── [2/4] ペアリスト生成（exhaustive） ───────────────────────────────
        print(f"[2/{total}] ペアリスト生成（exhaustive）...", flush=True)
        pairs_from_exhaustive.main(sfm_pairs, image_list=None, features=features_path)

    # ── [N-1/N] 局所特徴点マッチング ─────────────────────────────────────────
    match_step = total - 1
    print(f"[{match_step}/{total}] 特徴点マッチング: {args.matcher_type}", flush=True)
    matcher_conf = match_features.confs[args.matcher_type]
    match_features.main(matcher_conf, sfm_pairs,
                        features=features_path, matches=matches_path)

    # ── [N/N] SfM再構成 ───────────────────────────────────────────────────────
    print(f"[{total}/{total}] SfM再構成...", flush=True)
    model = reconstruction.main(
        sfm_dir, images, sfm_pairs, features_path, matches_path
    )

    if model is None:
        print("ERROR: 再構成に失敗しました。", file=sys.stderr)
        sys.exit(1)

    print(f"完了: {len(model.images)} カメラ, {len(model.points3D)} 点群", flush=True)
    print(f"出力先: {sfm_dir}", flush=True)

    # ── undistortion ─────────────────────────────────────────────────────────
    # HLocはSIMPLE_RADIAL等の歪みモデルを使うため、3DGS学習前にPINHOLEへ変換する
    print("[undistortion] PINHOLE変換を開始...", flush=True)
    dense = source / "dense"
    ret = subprocess.run([
        "colmap", "image_undistorter",
        "--image_path",  str(images),
        "--input_path",  str(sfm_dir),
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

    print(f"undistortion完了: {dense}", flush=True)


if __name__ == "__main__":
    main()

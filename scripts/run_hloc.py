# HLoc（SuperPoint/DISK/SIFT + LightGlue/SuperGlue等）を使ってカメラ姿勢推定を行うスクリプト
# 出力は COLMAP 互換形式（sparse/0/）で gaussian-splatting にそのまま渡せる
# ペアリスト生成方式: exhaustive（全ペア・4ステップ）または retrieval（類似画像のみ・5ステップ）

import argparse
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, "/opt/hloc")


class _TqdmFilter(io.TextIOBase):
    """tqdmのプログレスバー行を間引いてログサイズを抑えるフィルタ。
    連続するプログレスバー行は先頭2行・末尾1行のみ書き出し、中間は省略行に置き換える。
    """
    _PAT = re.compile(r"\d+%\|")

    def __init__(self, stream, keep_first=2, keep_last=1):
        self._stream  = stream
        self._keep_first = keep_first
        self._keep_last  = keep_last
        self._buf = []

    def _flush_buf(self):
        if not self._buf:
            return
        lines = self._buf
        out = lines[:self._keep_first]
        omitted = len(lines) - self._keep_first - self._keep_last
        if omitted > 0:
            out.append(f"  ... （中間 {omitted} 行省略）\n")
        if len(lines) > self._keep_first:
            out.extend(lines[-self._keep_last:])
        for l in out:
            self._stream.write(l)
        self._stream.flush()
        self._buf = []

    def write(self, s):
        for line in s.splitlines(keepends=True):
            clean = line.replace("\r", "")
            if not clean.endswith("\n"):
                clean += "\n"
            if self._PAT.search(clean):
                self._buf.append(clean)
            else:
                self._flush_buf()
                self._stream.write(clean)
        return len(s)

    def flush(self):
        self._flush_buf()
        self._stream.flush()


# stdout/stderr 両方にフィルタを適用（ログファイルへの書き出し量を削減）
sys.stdout = _TqdmFilter(sys.stdout)
sys.stderr = _TqdmFilter(sys.stderr)

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


if __name__ == "__main__":
    main()

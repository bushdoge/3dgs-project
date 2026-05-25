# HLoc 特徴点抽出器・マッチャーの選択肢と説明
# 01_pipeline.py / 03_colmap.py から共通インポートして使用する

FEATURE_OPTIONS = [
    "superpoint_max",
    "superpoint_aachen",
    "disk",
    "aliked-n16",
    "sift",
    "r2d2",
    "d2net-ss",
]

MATCHER_OPTIONS = [
    "superpoint+lightglue",
    "disk+lightglue",
    "aliked+lightglue",
    "superglue",
    "superglue-fast",
    "NN-superpoint",
    "NN-ratio",
    "NN-mutual",
    "adalam",
]

FEATURE_DESC = {
    "superpoint_max":    "SuperPoint（精度優先）— 高精度版。テクスチャが豊富な屋内外シーンに最適。やや重い。",
    "superpoint_aachen": "SuperPoint（標準）— バランス型。多くのシーンで安定して動作する定番の選択肢。",
    "disk":              "DISK — 繰り返し性が高く、テクスチャの少ない面やぼかしに強い学習ベース特徴点。",
    "aliked-n16":        "ALIKED-n16 — 軽量・高速な深層学習特徴点。LightGlue との相性が良くバランスに優れる。",
    "sift":              "SIFT — 古典的な特徴点アルゴリズム。高速だが深層学習ベースより複雑なシーンでは精度が落ちる。",
    "r2d2":              "R2D2 — 繰り返し性と信頼度を同時に学習。照明変化に強くテクスチャ豊富なシーンで有効。",
    "d2net-ss":          "D2-Net — 検出と記述を単一ネットワークで処理。低テクスチャ・反射面に強い。",
}

MATCHER_DESC = {
    "superpoint+lightglue": "SuperPoint 向け LightGlue — Transformer ベースの高精度マッチャー。SuperPoint と組み合わせて使用する推奨構成。",
    "disk+lightglue":        "DISK 向け LightGlue — DISK の特徴点に最適化された LightGlue。",
    "aliked+lightglue":      "ALIKED 向け LightGlue — 軽量・高速な組み合わせ。処理速度を重視する場合に有効。",
    "superglue":             "SuperGlue — Transformer 型の高精度マッチャー。精度は高いが処理が重い。",
    "superglue-fast":        "SuperGlue（高速版）— SuperGlue の軽量設定版。精度を少し落として処理を高速化。",
    "NN-superpoint":         "SuperPoint 用 最近傍マッチング — シンプルな最近傍探索。高速だが誤対応が増えやすい。",
    "NN-ratio":              "比率テスト付き最近傍マッチング — SIFT と組み合わせる古典的手法。高速・軽量。",
    "NN-mutual":             "相互最近傍マッチング — 双方向で最近傍を確認するシンプルな手法。堅実で安定。",
    "adalam":                "AdaLAM — 局所的な幾何制約を使うアダプティブなマッチャー。外れ値に強い。",
}

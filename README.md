# 3DGS Lab

**360度動画・通常動画・画像群から、高品質な 3D Gaussian Splatting（3DGS）を作るための実験プラットフォーム。**
フレーム抽出 → カメラ姿勢推定 → 学習 → 評価までを Streamlit GUI から一気通貫に操作できます。

---

## 特徴

- **360度動画にワンストップ対応** — 等距円筒フレームの抽出からピンホール多方向への変換、姿勢推定、学習まで自動化。
- **撮影者を自動で消す（SAM2マスク）** — 360度画像上で撮影者を数クリックするだけで、SAM2が全フレーム・全方向に追跡・投影し、学習から除外します。撮影者がカメラの周りを一周しても途切れません（[詳細](USAGE.md#sam2マスク撮影者の除去)）。
- **GUI から全部操作** — 設定をまとめてキューに積み、バックグラウンドで順次実行。ブラウザを閉じても進行します。
- **COLMAP / HLoc を選択可能** — 古典SfM（COLMAP）と学習ベース（HLoc: SuperPoint+LightGlue 等）を切り替え。
- **評価・比較が内蔵** — PSNR / SSIM / LPIPS を学習中に記録し、複数実験の学習曲線を重ねて比較。
- **点群クリーニング（SOR）** — 統計的外れ値除去でフローターの種になるノイズ点を除去。

---

## パイプライン概要

```
[入力] 360度動画 / 通常動画 / 画像フォルダ
   │
   ▼ フレーム抽出（360度は等距円筒→ピンホール変換）
[input/]  +  [equirect/]（360度のみ・マスク用）
   │
   ▼ カメラ姿勢推定（COLMAP / HLoc）
[sparse/0/]
   │
   ▼ SAM2マスク生成（任意・撮影者の除去）         ← 360度は等距円筒で1回指定→全方向へ投影
[masks/]
   │
   ▼ 3DGS学習（masks/ があれば撮影者を自動除外）
[output/]
   │
   ▼ レンダリング・評価（PSNR / SSIM / LPIPS）
[renders/]
```

各ステップは独立して実行でき、途中から再開も可能です。

---

## クイックスタート

```bash
# 1. 素材を置く（いずれか）
#   360度動画 → data/360movies/<name>.mp4
#   通常動画  → data/movies/<name>.mp4
#   画像群    → data/images/<scene>/*.jpg

# 2. GUI を起動
streamlit run /workspace/streamlit_app.py
# → ブラウザで http://localhost:8501

# 3. 「パイプライン」ページで入力・各ステップの設定をして実行
#    （撮影者を消したい場合は「SAM2マスク」ページを学習の前に実行）
```

> 詳しい手順・各ページの使い方・CLI からの実行方法は **[USAGE.md](USAGE.md)** を参照してください。

---

## 動作環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA（CUDA対応。検証環境は RTX A6000 / VRAM 48GB） |
| ベース | Docker（Ubuntu 22.04 + CUDA 12.2） |
| 主要依存 | PyTorch (cu121)、FFmpeg、COLMAP 3.9、HLoc、SAM2、Streamlit |
| 3DGS本体 | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |

セットアップ・依存の詳細は [USAGE.md](USAGE.md#セットアップ) を参照。

---

## 謝辞・ライセンス

本プロジェクトは学習・研究目的の実験用リポジトリで、以下の成果に依存しています。

- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)（3DGS本体・非商用研究ライセンス）
- [SAM 2](https://github.com/facebookresearch/sam2)（撮影者マスク生成）
- [HLoc](https://github.com/cvg/Hierarchical-Localization) / [COLMAP](https://colmap.github.io/)（カメラ姿勢推定）

各依存ライブラリのライセンス条件に従ってください。

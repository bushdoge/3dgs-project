# CLAUDE.md — 3DGS実験プロジェクト

このファイルはAIアシスタント（Claude）向けのプロジェクト説明・ルール集です。

---

## プロジェクト概要

360度動画・通常動画・画像群を入力とし、高品質な3D Gaussian Splatting（3DGS）を生成するための学習・実験用リポジトリです。

---

## 環境

| 項目 | 内容 |
|---|---|
| 接続構成 | 自宅PC → SSH → 踏み台サーバ → 研究室GPUマシン |
| コンテナ | Docker（Linux上で稼働） |
| GPU | NVIDIA RTX A6000（VRAM 48GB） |
| Python | 3.10（システムPython） |
| PyTorch | 2.5.1 + CUDA 12.1 |
| 3DGSフレームワーク | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| 前処理 | FFmpeg 4.4.2、COLMAP 3.9（CUDA対応ビルド）、HLoc 1.5 |
| UI | Streamlit、noVNC |

---

## ディレクトリ構造

```
/workspace/
├── CLAUDE.md               # このファイル
├── streamlit_app.py        # GPUモニター（Streamlit）
├── data/                   # 元動画・元画像置き場（削除・git add 厳禁）
│   └── <scene_name>/       # シーンごとに分ける
│       ├── video.mp4       # 元動画
│       └── images/         # 元画像群
├── experiments/            # 実験結果
│   └── YYYYMMDD_HHMMSS_<scene_name>/   # 日時+シーン名
│       ├── config.yaml     # 実験設定
│       ├── frames/         # 切り出した連番画像
│       ├── colmap/         # COLMAP出力（sparse/ dense/ 等）
│       ├── output/         # 3DGS学習結果（point_cloud等）
│       ├── renders/        # レンダリング結果
│       ├── logs/           # 学習ログ
│       └── source_video -> /workspace/data/<scene_name>/video.mp4  # シンボリックリンク
├── scripts/                # 各種処理スクリプト
│   ├── extract_frames.py       # 動画→連番画像の切り出し
│   ├── convert_360.py          # 360度動画のピンホール変換
│   ├── run_colmap.py           # COLMAP実行ラッパー
│   ├── run_train.py            # 3DGS学習実行ラッパー
│   └── pipeline.py             # パイプライン統合スクリプト（GUI対応）
├── tmp/                    # 一時作業ファイル置き場（gitignore推奨）
└── /opt/gaussian-splatting/    # 公式ソース（読み取り専用・直接変更禁止）
```

> シーン名は元動画のファイル名または画像フォルダ名から自動で取得します。

---

## パイプライン

以下のステップで構成されます。**任意のステップから開始・終了できます（スキップ可能）。**

```
[Step 1] 動画/画像の入力
    ↓
[Step 2] フレーム抽出（FFmpeg）
    ↓
[Step 3] カメラ姿勢推定（COLMAP / HLoc）
    ↓
[Step 4] 3DGS学習（gaussian-splatting / train.py）
    ↓
[Step 5] 評価・可視化・レンダリング
```

- **360度動画**：ピンホール変換（Step 2前に実施）と等距円筒そのまま利用（360GS等）の両方を試す予定。
- **Reality Scan出力**：COLMAPをスキップしてStep 4から開始できるようにする。
- パイプラインの制御は**Streamlit GUI（優先）またはコマンドライン引数**で行う。
- GUIにはnoVNCおよびStreamlitを活用する。

---

## Claudeへのルール（必ず守ること）

### コミュニケーション
- **やり取りはすべて日本語**で行うこと。

### ファイル操作
- `data/` 配下のファイルは**絶対に削除しない**こと（元動画・元画像が入っている）。
- `storage/` 配下のフォルダ・ファイルは**変更・削除しない**こと。
- `/opt/gaussian-splatting/` の**ソースコードを直接書き換えない**こと。変更が必要な場合は `/workspace/` にコピーしてから行うこと。
- 一時的な作業ファイルは **`/workspace/tmp/`** にまとめること。
- 大きなファイルはコピーせず、**シンボリックリンクや元パスの参照**で対応してストレージを節約すること。

### Git操作
- `.gitignore` のルールを**厳守**すること。
- **`data/` 配下のファイルを絶対に `git add` しない**こと。
- コミットメッセージは**何を修正・追加したか具体的に日本語**で書くこと。

### 実行・処理
- `train.py` などの**長時間処理やGPUを占有する処理を開始する前は必ず確認**すること。

### コーディング
- 各スクリプトの**冒頭にそのファイルが何をするコードなのかをコメントで記載**すること。
- フォルダ構造は多少深くなっても**わかりやすさを優先**すること。

---

## よく使うコマンド

```bash
# GPUモニター起動
streamlit run /workspace/streamlit_app.py

# フレーム抽出（例）
python scripts/extract_frames.py --input data/scene1/video.mp4 --output experiments/YYYYMMDD_HHMMSS_scene1/frames/

# COLMAP実行（例）
python scripts/run_colmap.py --image_path experiments/YYYYMMDD_HHMMSS_scene1/frames/

# 3DGS学習（例・実行前に必ず確認）
python scripts/run_train.py --source experiments/YYYYMMDD_HHMMSS_scene1/
```

---

## 注意事項・補足

- ユーザーはGitHub・Docker・Claudeの初心者です。専門用語は丁寧に説明してください。
- Docker・Git操作に関する質問が多い可能性があります。わかりやすく答えてください。
- HLocは `/opt/hloc/` にインストール済みです（バージョン 1.5）。
- COLMAP は 3.9（CUDA対応ビルド）に更新済みです。

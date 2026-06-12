# CLAUDE.md — 3DGS実験プロジェクト

---

## プロジェクト概要

360度動画・通常動画・画像群を入力とし、高品質な3D Gaussian Splatting（3DGS）を生成するための学習・実験用リポジトリです。

---

## 環境

| 項目 | 内容 |
|---|---|
| 接続構成 | 自宅PC → SSH → 踏み台サーバ → 研究室GPUマシン |
| ベースイメージ | `nvidia/cuda:12.2.0-devel-ubuntu22.04` |
| コンテナ | Docker（Ubuntu 22.04 + CUDA 12.2） |
| GPU | NVIDIA RTX A6000（VRAM 48GB、CUDA Arch 8.6） |
| Python | 3.10（Ubuntu 22.04 システムPython） |
| PyTorch | cu121 wheels（`--index-url https://download.pytorch.org/whl/cu121`） |
| 3DGSフレームワーク | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)（`/opt/gaussian-splatting/`） |
| 前処理 | FFmpeg（apt）、COLMAP 3.9（CUDA対応ソースビルド）、HLoc（最新 main） |
| UI | Streamlit（ポート 8501）、noVNC（ポート 6080） |
| 開発ツール | Node.js 20、Claude Code CLI（`@anthropic-ai/claude-code`） |

---

## ディレクトリ構造

```
/workspace/
├── CLAUDE.md               # このファイル
├── streamlit_app.py        # ホーム画面（ToDo・使用方法・ナビゲーション）
├── data/                   # 元素材置き場（削除・git add 厳禁）
│   ├── 360movies/          # 360度動画（例: garden.mp4）
│   ├── movies/             # 通常動画（例: desk.mp4）
│   └── images/             # 画像群（シーン名サブフォルダ必須）
│       └── <scene_name>/   # 例: images/garden/001.jpg
├── experiments/            # 実験結果
│   └── YYYYMMDD_<scene_name>_NN/   # 日付+シーン名+連番（例: 20260429_Ylab_room_v2_mid_01）
│       ├── pipeline_config.json    # 実験設定（パイプライン/バッチ実行時に自動保存）
│       ├── input/          # 切り出した連番画像（COLMAP/HLocへの入力）
│       ├── sparse/0/       # 姿勢推定結果（COLMAP互換モデル）
│       ├── dense/          # undistortion後の画像+モデル（PINHOLE以外のとき生成）
│       ├── hloc_outputs/   # HLocの中間生成物（features.h5等）
│       ├── masks/          # SAM2撮影者マスク（任意。あると学習時に自動合成）
│       ├── output/         # 3DGS学習結果（point_cloud等）
│       ├── extract_log.txt / colmap_log.txt   # 各ステップのログ
│       └── note.md         # 自由メモ（気づき・失敗原因など）
├── models/                 # 事前学習済みモデル置き場
│   └── pretrained/         # 外部からDLしたモデルファイル（SAM2チェックポイント等）
├── queue_helper.py         # バッチキュー共通ユーティリティ（flockで排他）
├── job_commands.py         # バッチジョブのコマンド構築（デーモン/GUI共通）
├── pipeline_widget.py      # 進捗ウィジェット共通コンポーネント
├── hloc_options.py         # HLoc特徴点・マッチャー選択肢の定義
├── pages/                  # Streamlitページ（00_batch〜91_monitor）
├── scripts/                # 各種処理スクリプト
│   ├── extract_frames.py       # 動画→連番画像の切り出し
│   ├── convert_360.py          # 360度動画のピンホール変換
│   ├── run_colmap.py           # COLMAP実行ラッパー
│   ├── run_hloc.py             # HLoc実行ラッパー（COLMAP互換出力）
│   ├── run_train.py            # 3DGS学習実行ラッパー（マスク合成・自動undistortion対応）
│   ├── train_custom.py         # train.py改変版（SSIM/LPIPS評価ログ追加）
│   ├── run_render.py           # レンダリング実行ラッパー
│   ├── generate_masks.py       # SAM2撮影者マスク生成 + SOR点群クリーニング
│   ├── batch_daemon.py         # バッチキュー自動実行デーモン
│   └── recover_pipeline_config.py  # pipeline_config.json復元ツール
├── tmp/                    # 一時作業ファイル置き場（gitignore推奨）
└── /opt/gaussian-splatting/    # 公式ソース（読み取り専用・直接変更禁止）
```

> シーン名は動画のファイル名（拡張子なし）または画像フォルダ名から自動で取得します。
> 入力タイプの対応：360度動画 → `data/360movies/`、通常動画 → `data/movies/`、画像群 → `data/images/<scene_name>/`

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

## 開発規約

### ファイル操作
- `data/` 配下のファイルは**絶対に削除しない**こと（元動画・元画像が入っている）。
- `storage/` 配下のフォルダ・ファイルは**変更・削除しない**こと。
- `/opt/gaussian-splatting/` の**ソースコードを直接書き換えない**こと。変更が必要な場合は `/workspace/` にコピーしてから行うこと。
- 一時的な作業ファイルは **`/workspace/tmp/`** にまとめること。
- 大きなファイルはコピーせず、**シンボリックリンクや元パスの参照**で対応してストレージを節約すること。
- **Dockerfileはホストマシン側にあるため、Docker内から直接編集・変更することはできない**。Dockerfileの変更が必要な場合は、変更箇所をユーザーに伝えて、ユーザー自身に編集してもらうこと。

### Git操作
- `.gitignore` のルールを**厳守**すること。
- **`data/` 配下のファイルを絶対に `git add` しない**こと。
- コミットメッセージは**何を修正・追加したか具体的に日本語**で書くこと。
- **コミット・プッシュはユーザーから明示的に指示があったときのみ行うこと。**

### コーディング
- 各スクリプトの**冒頭にそのファイルが何をするコードなのかをコメントで記載**すること。
- フォルダ構造は多少深くなっても**わかりやすさを優先**すること。

---

## よく使うコマンド

```bash
# Streamlit GUI起動
streamlit run /workspace/streamlit_app.py

# フレーム抽出（例）
python scripts/extract_frames.py --input data/movies/scene1.mp4 --output experiments/YYYYMMDD_scene1_01/input/

# COLMAP実行（例）
python scripts/run_colmap.py --source_path experiments/YYYYMMDD_scene1_01/

# HLoc実行（例）
python scripts/run_hloc.py --source_path experiments/YYYYMMDD_scene1_01/

# 3DGS学習（例）
python scripts/run_train.py --source experiments/YYYYMMDD_scene1_01/ --model_path experiments/YYYYMMDD_scene1_01/output/

# バッチデーモン起動（ブラウザなしでキューを自動実行）
nohup python3 scripts/batch_daemon.py > /dev/null 2>&1 &
```

---

## 補足

- HLocは `/opt/hloc/` にインストール済みです（バージョン 1.5）。
- COLMAP は 3.9（CUDA対応ビルド）に更新済みです。

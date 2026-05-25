# 3DGS Lab

360度動画・通常動画・画像群を入力として 3D Gaussian Splatting（3DGS）を生成する実験用プロジェクトです。  
Streamlit GUI からパイプライン全体を操作できます。

---

## アプリ構成

`streamlit_app.py` がエントリーポイントで、`pages/` 配下のページを `st.navigation` で束ねています。

```
streamlit_app.py
└── pages/
    ├── 90_home.py             # ホーム・ToDo・パイプライン進捗確認
    ├── 91_monitor.py          # GPU / CPU / メモリのリアルタイム監視
    │
    ├── 00_batch.py            # バッチキュー管理・ジョブ実行
    ├── 01_pipeline.py         # 実験設定 → キューに追加
    ├── 02_frame_extraction.py # フレーム抽出（360度変換オプション付き）
    ├── 03_colmap.py           # カメラ姿勢推定（COLMAP / HLoc）
    ├── 04_training.py         # 3DGS 学習・リアルタイムログ表示
    ├── 05_results.py          # レンダリング実行・結果確認
    ├── 06_compare.py          # 複数実験の Loss / PSNR 比較
    └── 07_experiment_manager.py # 実験一覧・削除・メモ管理
```

### 補助モジュール

| ファイル | 役割 |
|---|---|
| `queue_helper.py` | バッチキューの読み書き（JSON ベース）を共通化 |
| `pipeline_widget.py` | 各ページに埋め込む「パイプライン進捗ウィジェット」 |
| `hloc_options.py` | HLoc の特徴量抽出・マッチャー設定を一元管理 |

---

## パイプライン

```
[入力] 360度動画 / 通常動画 / 画像フォルダ
    ↓
[Step 1] フレーム抽出          extract_frames.py  （FFmpeg）
    ↓                          ※360度動画は convert_360.py でピンホール変換
[Step 2] カメラ姿勢推定        run_colmap.py / run_hloc.py  （Structure from Motion）
    ↓
[Step 3] 3DGS 学習             run_train.py  （gaussian-splatting/train.py のラッパー）
    ↓
[Step 4] レンダリング・評価    run_render.py
```

各ステップは独立して実行でき、途中から再開することも可能です。  
01_pipeline.py からすべての設定をまとめてキューに追加するワンストップフローが標準の使い方です。

---

## スクリプト詳細（`scripts/`）

### `extract_frames.py`
FFmpeg で動画から連番画像を切り出します。進捗を `PROGRESS cur/total` 形式で stdout に出力し、Streamlit 側でプログレスバーに反映します。

### `convert_360.py`
等距円筒（Equirectangular）画像をピンホールカメラ視点の画像群に変換します。前・後・左・右・上・下の最大6方向を指定方向角度ごとにクロップして出力します。

### `run_colmap.py`
COLMAP による SfM を4ステップ（特徴点抽出 → マッチング → マッパー → undistortion）に分けて実行するラッパーです。`[COLMAP N/4]` 形式のマーカーで進捗を通知します。

### `run_hloc.py`
HLoc（SuperPoint / DISK / SIFT + LightGlue / SuperGlue など）で SfM を実行します。ペアリスト生成方式を **exhaustive**（全ペア）と **retrieval**（類似画像のみ）から選択できます。出力は COLMAP 互換形式（`sparse/0/`）なので gaussian-splatting にそのまま渡せます。

### `run_train.py`
`/opt/gaussian-splatting/train.py` を実験ディレクトリを指定して呼び出すラッパーです。カメラモデルが PINHOLE 以外の場合は自動で undistortion を挟みます。VRAM と画像サイズから縮小解像度を自動計算する機能もあります。

### `run_render.py`
学習済みモデルから全カメラ視点の画像をレンダリングします。結果は `renders/` に保存されます。

### `generate_masks.py`（`feature/sam2` ブランチ）
SAM2（SAM 2.1 Hiera-Large）を使って画像内の撮影者をマスクします。クリック座標を JSON で指定すると、360度動画由来の方向別フレーム群それぞれに対してポイントプロンプトを時系列に伝播させ、`masks/` フォルダに保存します。生成したマスクは学習時に撮影者を除外するために使用します。

### `batch_daemon.py`
バッチキューを監視してジョブを順次実行するデーモンです。Streamlit を開いていなくても動き続けます。`nohup python3 scripts/batch_daemon.py &` またはキューページの起動ボタンから開始できます。

---

## 実験管理

実験ごとに `experiments/YYYYMMDD_<scene_name>/` フォルダが作成され、以下の構成で出力が保存されます。

```
experiments/YYYYMMDD_<scene>/
├── config.yaml       # 実験設定（パイプライン全ステップの引数）
├── frames/           # 切り出した連番画像
├── colmap/           # COLMAP / HLoc 出力（sparse/ 等）
├── input/            # 学習に渡す画像（undistorted）
├── masks/            # SAM2 マスク画像
├── output/           # 3DGS 学習結果（point_cloud 等）
├── renders/          # レンダリング画像
├── logs/             # 学習ログ（loss_curve.json 等）
└── note.md           # 自由メモ
```

`07_experiment_manager.py` から全実験の状態・ディスク使用量・メモをまとめて確認・編集できます。  
`06_compare.py` では複数実験の Loss / PSNR 学習曲線を重ね表示して性能比較ができます。

---

## ブランチ構成

| ブランチ | 内容 |
|---|---|
| `main` | SAM2 実装前の安定版 |
| `develop` | 開発統合ブランチ |
| `feature/sam2` | SAM2 マスク生成と学習統合の実装 |

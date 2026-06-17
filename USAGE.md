# USAGE — 使い方ガイド

3DGS Lab の詳細な使い方です。概要は [README.md](README.md) を参照してください。

## 目次

- [セットアップ](#セットアップ)
- [入力データの置き方](#入力データの置き方)
- [GUIの起動とページ構成](#guiの起動とページ構成)
- [標準ワークフロー](#標準ワークフロー)
- [SAM2マスク（撮影者の除去）](#sam2マスク撮影者の除去)
- [CLIから実行する](#cliから実行する)
- [スクリプト詳細](#スクリプト詳細)
- [実験フォルダの構成](#実験フォルダの構成)
- [バッチ実行・デーモン](#バッチ実行デーモン)
- [既知の制約・Tips](#既知の制約tips)

---

## セットアップ

Docker（Ubuntu 22.04 + CUDA 12.2）上で動作します。主要な依存は次のとおりです。

| 種別 | 内容 |
|---|---|
| 3DGS本体 | [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)（`/opt/gaussian-splatting/`） |
| 前処理 | FFmpeg、COLMAP 3.9（CUDA対応）、HLoc |
| マスク | SAM2（SAM 2.1 Hiera-Large。チェックポイントは `models/pretrained/sam2/`） |
| 評価 | lpips（LPIPS指標） |
| UI | Streamlit（ポート8501）、画像クリック用 `streamlit-image-coordinates` |

> SAM2 / lpips / streamlit-image-coordinates などは Docker イメージに含めておくと、
> コンテナ再構築時に消えません。

---

## 入力データの置き方

入力タイプごとに置き場所が決まっています。**シーン名はファイル名／フォルダ名から自動取得**されます。

```
data/
├── 360movies/          # 360度動画（例: garden.mp4）
├── movies/             # 通常動画（例: desk.mp4）
└── images/
    └── <scene_name>/   # 画像群（シーン名サブフォルダ必須。例: images/garden/001.jpg）
```

> `data/` 配下は元素材です。削除や `git add` をしないでください。

---

## GUIの起動とページ構成

```bash
streamlit run /workspace/streamlit_app.py
# → http://localhost:8501
```

`streamlit_app.py` がエントリーポイントで、`pages/` を `st.navigation` で束ねています。

| ページ | 役割 |
|---|---|
| ホーム（`90_home.py`） | ToDo・パイプライン進捗の確認 |
| システムモニター（`91_monitor.py`） | GPU / CPU / メモリのリアルタイム監視 |
| キュー（`00_batch.py`） | バッチキューの管理・ジョブ実行 |
| パイプライン（`01_pipeline.py`） | 実験設定をまとめてキューに追加（標準の入口） |
| フレーム抽出（`02_frame_extraction.py`） | 動画→連番画像（360度変換オプション付き） |
| 姿勢推定（`03_colmap.py`） | カメラ姿勢推定（COLMAP / HLoc） |
| SAM2マスク（`08_sam2_masks.py`） | 撮影者マスク生成・SOR点群クリーニング |
| 3DGS学習（`04_training.py`） | 学習・リアルタイムログ表示 |
| 結果確認（`05_results.py`） | レンダリング・結果確認 |
| 比較（`06_compare.py`） | 複数実験の Loss / PSNR 学習曲線を重ね表示 |
| 実験マネージャ（`07_experiment_manager.py`） | 実験一覧・ディスク使用量・削除・メモ管理 |

---

## 標準ワークフロー

撮影者を消したい場合、**SAM2マスクは姿勢推定の後・学習の前**に実行します
（バッチの「パイプライン」は抽出→姿勢推定→学習を一気に流すため、撮影者を消すなら下記の順で手動実行します）。

```
[02] フレーム抽出（360度）   ← input/ と equirect/ が生成される
        ↓
[03] カメラ姿勢推定（COLMAP / HLoc）
        ↓
[08] SAM2マスク生成（等距円筒モード）   ← 学習より前に必須
        ↓
[04] 3DGS学習                ← masks/ を自動検出して撮影者を除外
        ↓
[05] レンダリング・結果確認
```

撮影者を消す必要がなければ、「パイプライン」ページで設定をまとめてキューに積み、抽出→姿勢推定→学習を自動実行するのが最短です。

---

## SAM2マスク（撮影者の除去）

360度動画で自分が映り込んでしまった撮影者を、学習から自動で除外します。

### 仕組み

360度動画の抽出時に保存される**等距円筒フレーム（`equirect/`）**に対して SAM2 を1系統だけ実行し、
出来たマスクを**画像変換と同じ幾何変換でピンホール全方向に投影**します。これにより、

- 撮影者の指定は **360度画像上で1回クリックするだけ**（方向ごとに指定する必要がない）
- 画像とマスクのピクセル対応が厳密に一致
- 撮影者がカメラの周りを**一周移動しても途切れない**（継ぎ目対策の2パス方式）

### 手順（GUI）

1. 「SAM2マスク」ページで実験を選択
2. `equirect/` がある実験では「等距円筒（推奨）／方向別」のモード選択が出るので **等距円筒** を選ぶ
   - 出てこない場合は `equirect/` が無いので、**フレーム抽出（02）からやり直し**が必要
3. 表示された360度画像で、**撮影者の上を1〜3点クリック**（撮影者点／背景抑制点を切り替え可能）
   - 1枚目に撮影者が写っていなければ、スライダーで写っているフレームに移動してからクリック
4. 必要なら「マスク膨張(px)」を調整（既定7でだいたいOK）
5. 「マスク生成を実行」 → 全方向の `masks/` が生成される
6. プレビューで除外領域を確認。ずれていれば点を足して再実行
7. そのまま「3DGS学習」を実行すると `masks/` が自動検出され、撮影者が除外される

### モードの違い

| モード | 入力 | 指定回数 | 用途 |
|---|---|---|---|
| 等距円筒 | `equirect/` の360度画像 | シーンにつき1系統 | 360度動画（推奨） |
| 方向別 | `input/` のピンホール画像 | 方向ごと | `equirect/` の無い実験・通常動画・画像群 |

### SOR（点群クリーニング）

COLMAP点群の統計的外れ値を除去します。`sparse/0/` と `dense/sparse/0/` の `points3D.bin` を
直接書き換え（元モデルは `before_sor/` にバックアップ）、次回の学習から除去後の点群が初期値になります。
マスク生成と同時、または単独（`--sor-only`）で実行できます。

---

## CLIから実行する

GUI を使わずコマンドラインからも実行できます。

```bash
# フレーム抽出（通常動画）
python scripts/extract_frames.py --input data/movies/scene1.mp4 \
  --output experiments/YYYYMMDD_scene1_01/input/

# フレーム抽出（360度動画 → ピンホール変換 + 等距円筒も保存）
python scripts/convert_360.py --input data/360movies/scene.mp4 \
  --output experiments/YYYYMMDD_scene_01/input/ \
  --fov 90 --width 1024 --height 1024 --fps 1.0 \
  --angles 0,0 90,0 180,0 270,0 \
  --keep-equirect experiments/YYYYMMDD_scene_01/equirect/

# 姿勢推定（COLMAP / HLoc いずれか）
python scripts/run_colmap.py --source_path experiments/YYYYMMDD_scene_01/
python scripts/run_hloc.py   --source_path experiments/YYYYMMDD_scene_01/

# SAM2マスク（等距円筒モード）
python scripts/generate_masks.py experiments/YYYYMMDD_scene_01/ --sam-only --equirect \
  --clicks-json '{"equirect": {"frame": 0, "points": [[1024, 900, 1]]}}'

# 3DGS学習（masks/ があれば自動で撮影者を除外）
python scripts/run_train.py --source experiments/YYYYMMDD_scene_01/ \
  --model_path experiments/YYYYMMDD_scene_01/output/

# レンダリング
python scripts/run_render.py -m experiments/YYYYMMDD_scene_01/output/ \
  -s experiments/YYYYMMDD_scene_01/
```

---

## スクリプト詳細

`scripts/` 配下の各スクリプトの役割です。

### `extract_frames.py`
FFmpeg で動画から連番画像を切り出します。進捗を `PROGRESS cur/total` 形式で stdout に出力し、GUIのプログレスバーに反映します。

### `convert_360.py`
等距円筒（Equirectangular）画像を指定方向角度（yaw, pitch）ごとにピンホール視点へクロップします。`--keep-equirect <dir>` を付けると、変換元の等距円筒フレームと変換パラメータ（`meta.json`）も保存します（SAM2の等距円筒モードで使用）。

### `run_colmap.py`
COLMAP による SfM を4ステップ（特徴点抽出 → マッチング → マッパー → undistortion）に分けて実行するラッパー。`[COLMAP N/4]` 形式で進捗を通知します。

### `run_hloc.py`
HLoc（SuperPoint / DISK / SIFT + LightGlue / SuperGlue など）で SfM を実行。ペアリスト生成を **exhaustive**（全ペア）と **retrieval**（類似画像のみ）から選択できます。出力は COLMAP 互換（`sparse/0/`）。

### `run_train.py`
学習スクリプト（`train_custom.py`）を実験ディレクトリ指定で呼び出すラッパー。カメラモデルが PINHOLE 以外なら自動で undistortion を挟み、VRAMと画像サイズから縮小解像度を自動計算します。`masks/` があればアルファとして学習画像に合成し（undistortion済みならマスクも再マップ）、撮影者領域を学習から除外します。

### `train_custom.py`
gaussian-splatting の `train.py` 改変版。学習中に SSIM / LPIPS / PSNR を記録します。

### `run_render.py`
学習済みモデルから全カメラ視点をレンダリングし、`renders/` に保存します。

### `generate_masks.py`
SAM2（SAM 2.1 Hiera-Large）で撮影者をマスクします。

- **方向別モード**：`input/` のピンホール画像に、方向ごとにクリック点を与えて時系列伝播。
- **等距円筒モード（`--equirect`）**：`equirect/` の360度フレームに1系統実行し、`meta.json` の変換でピンホール全方向に投影。継ぎ目対策の2パスunionで撮影者の全周移動に対応。
- **SOR**：COLMAP点群の外れ値除去。`--sor-only` / `--sam-only` で個別実行可。

### `batch_daemon.py`
バッチキューを監視してジョブを順次実行するデーモン。Streamlit を開いていなくても動き続けます。`nohup python3 scripts/batch_daemon.py &` またはキューページの起動ボタンから開始します。

---

## 実験フォルダの構成

実験ごとに `experiments/YYYYMMDD_<scene_name>_NN/` が作られます。

```
experiments/YYYYMMDD_<scene>_NN/
├── pipeline_config.json   # 実験設定（パイプライン全ステップの引数）
├── input/                 # 切り出した連番画像（姿勢推定への入力）
├── equirect/              # 等距円筒フレーム + meta.json（360度のみ・マスク用）
├── sparse/0/              # COLMAP / HLoc の姿勢推定結果（COLMAP互換モデル）
├── dense/                 # undistortion 後の画像 + モデル（PINHOLE以外のとき生成）
├── hloc_outputs/          # HLoc の中間生成物（features.h5 等）
├── masks/                 # SAM2マスク（任意。あると学習時に自動合成）
├── masks_equirect/        # 等距円筒モードのSAM2出力（masks/ への投影元）
├── output/                # 3DGS学習結果（point_cloud / train_log.txt 等）
├── extract_log.txt / colmap_log.txt   # 各ステップのログ
└── note.md                # 自由メモ
```

実験マネージャ（`07_experiment_manager.py`）で全実験の状態・ディスク使用量・メモを確認・編集でき、
比較ページ（`06_compare.py`）で複数実験の Loss / PSNR 学習曲線を重ねて性能比較できます。

---

## バッチ実行・デーモン

パイプラインページで実験設定をキューに追加すると、バックグラウンドのデーモンが順次実行します。

- デーモンはキュー追加時に自動起動されます（手動起動は `nohup python3 scripts/batch_daemon.py &`）。
- 稼働状態はキューページ上部の稼働中／停止中表示で確認できます。
- ライブラリを更新した後は Streamlit を再起動してください（稼働中プロセスの import が古いままになるため）。

---

## 既知の制約・Tips

- **既存実験は等距円筒モードに使えない**：`equirect/` は `--keep-equirect` 付きで抽出した実験にしかありません。過去の実験で使うにはフレーム抽出からやり直しが必要です。
- **パイプライン一括実行はマスクを挟まない**：撮影者を消すなら 02→03 を実行 → 08 でマスク → 04 学習、の順で進めてください。
- **マスクの効果検証**：撮影者は画素の数%しか占めず、3DGSの実行ごとのばらつき（±0.2 PSNR程度）に埋もれます。効果を見たいときは撮影者を含まない視点での評価や、レンダリングの目視（近接フローターの減少）で確認するのが確実です。
- **白壁・テクスチャの薄い面**：多視点拘束が効きにくくフローターが出やすい領域です。フレーム数・露出の安定が効きます。

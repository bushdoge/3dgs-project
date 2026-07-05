# wirebench — 細線構造×3DGS ベンチマーク

3DGSが電線・フェンス等の細い構造物を再構成できない問題を定量するための実験コード。
研究の背景・結果・今後の計画は `/workspace/memo/thin_structure_research.md` を参照（ローカル専用メモ）。

## 構成

| ファイル | 役割 |
|---|---|
| `make_scene.py` | Blenderで合成シーン（電線3本・フェンス・電柱・建物）を構築し、周回視点の学習画像＋GT一式をレンダリング。`--wire-radius`（電線半径m）、`--frames`（視点数）、`--enclosed`（空なし統制版）で条件を統制 |
| `analyze_sfm.py` | COLMAP点群をGT座標にUmeyama位置合わせし、細線上のSfM点の欠乏を定量。厳密版=電線スパンの**表面**から3cm以内（中心線距離−線半径で判定。半径はgt/scene_params.jsonから自動取得） |
| `eval_wires.py` | 学習済み3DGSのテストレンダを 全体/細線領域/背景 PSNR に分解して評価し、比較図を出力 |
| `run_sweep.py` | 上記＋COLMAP＋学習＋レンダを条件グリッドで一括実行するオーケストレータ。結果は `sweep_results.csv` に1行/条件で追記 |
| `sweep_results.csv` | スイープ結果の集計表（実験名・条件・SfM点数・PSNR分解・落差dB） |
| `figs/` | スイープ結果の図 |

## スイープ実行（推奨。以下の手動手順を全条件ぶん自動化したもの）

```bash
cd /workspace
nohup python3 research/wirebench/run_sweep.py \
    --radii 0.008 0.015 0.03 0.06 --frames 80 --variants open enclosed --iters 7000 \
    > tmp/sweep.log 2>&1 &
```

- 各ステージは出力があればスキップされるので、**中断しても同じコマンドで続きから再開できる**
  （日をまたいで再開する場合は `--date YYYYMMDD` に初回実行日を渡すこと。実験名が変わると再開できない）
- GPUが消えた場合は分かるメッセージで exit 2 で止まる → docker restart 後に同じコマンドを再実行
- 個別実験の詳細ログは `experiments/<実験名>/sweep.log`

## パイプライン（1実験の再現手順）

```bash
BLENDER=/workspace/tools/blender-4.2.5-linux-x64/blender
EXP=/workspace/experiments/YYYYMMDD_wirebench_XX

# 1. シーン生成＋レンダリング（CPU約10分/GPU約4分。--enclosed で空なし統制版）
$BLENDER -b -P make_scene.py -- --out $EXP --frames 80 --samples 64

# 2. SfM（合成データは歪みなしなので PINHOLE）
python3 /workspace/scripts/run_colmap.py --source_path $EXP --camera_model PINHOLE

# 3. SfM点群の細線欠乏を定量（[厳密]行が本命の数値）
cd $EXP && python3 /workspace/research/wirebench/analyze_sfm.py .

# 4. 3DGS学習（--eval で8枚おきにテスト分割）
python3 /workspace/scripts/run_train.py --source $EXP --model_path $EXP/output \
  --iterations 7000 --save_iterations 7000 --test_iterations 7000 --eval

# 5. テスト視点レンダ → 細線評価
python3 /workspace/scripts/run_render.py -m $EXP/output -s $EXP --iteration 7000 --skip_train
cd $EXP && python3 /workspace/research/wirebench/eval_wires.py . 7000
# 30k版を別フォルダで学習した場合: eval_wires.py . 30000 output_30k
```

## 出力（GT一式は `$EXP/gt/`）

- `poses.json` … GTカメラ（cam2world 4x4、Blender/OpenGL規約: 視線-Z・上+Y）＋内部パラメータ
- `mask_thin/NNNN.png` … 細線マスク（白=電線+フェンス棒。pass_index=1のIDMask）
- `depth/NNNN.exr` … 深度（float）
- `wire_points.npz` … 細線の3D点（points: (N,3) world座標 / labels: 1=電線 2=フェンス棒）
- `scene_params.json` … シーン生成条件（wire_radius・frames・enclosed等。分析側が参照する）
- `sfm_analysis.json` … analyze_sfm.py の集計値（run_sweep.pyが読む）
- `wire_eval_*/summary.json` … eval_wires.py の集計値（run_sweep.pyが読む）
- `sfm_points_overlay.png` … SfM点群を視点1に投影した可視化（analyze_sfm.pyが生成）
- `wire_eval_*/` … 比較図（GT|render、拡大クロップ。eval_wires.pyが生成）

## 実装上の注意（ハマりどころ）

- **カラーマネジメント**: AgX(デフォルト)は彩度が潰れSfMに不利、Standardは白飛びする → Filmicを使用
- **テクスチャ**: Brickテクスチャは垂直壁で縞になる → ノイズ系を使用。テクスチャ座標はObject（ワールド）出力
- **--enclosed**: 円筒プリミティブはデフォルトで蓋付き → 底面が地面とZファイトして真っ黒になる
  → `end_fill_type="NOTHING"` にしてある（天井は別の円盤）
- **電柱座標**: analyze_sfm.py の `POLE_XY` は make_scene.py の `pole_positions` と一致させること
- **テスト分割**: gaussian-splatting の `--eval` は名前順の8枚おき（llffhold=8）。
  eval_wires.py はこの規約でレンダ番号→フレーム名を対応させている

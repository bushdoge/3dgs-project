# wirebench — 細線構造×3DGS ベンチマーク

3DGSが電線・フェンス等の細い構造物を再構成できない問題を定量するための実験コード。
研究の背景・結果・今後の計画は `/workspace/memo/thin_structure_research.md` を参照（ローカル専用メモ）。

## 構成

| ファイル | 役割 |
|---|---|
| `make_scene.py` | Blenderで合成シーン（電線3本・フェンス・電柱・建物）を構築し、周回80視点の学習画像＋GT一式をレンダリング |
| `analyze_sfm.py` | COLMAP点群をGT座標にUmeyama位置合わせし、細線上のSfM点の欠乏を定量（厳密版=電線スパン3cm以内） |
| `eval_wires.py` | 学習済み3DGSのテストレンダを 全体/細線領域/背景 PSNR に分解して評価し、比較図を出力 |

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

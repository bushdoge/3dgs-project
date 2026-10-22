# wirebench — 細線構造×3DGS ベンチマーク

3DGSが電線・フェンス等の細い構造物を再構成できない問題を定量するための実験コード。
研究の背景・結果・今後の計画は `/workspace/memo/thin_structure_research.md` を参照（ローカル専用メモ）。

## 構成

| ファイル | 役割 |
|---|---|
| `make_scene.py` | Blenderで合成シーン（電線3本・フェンス・電柱・建物）を構築し、周回視点の学習画像＋GT一式をレンダリング。`--wire-radius`（電線半径m）、`--frames`（視点数）、`--enclosed`（空なし統制版）、`--layout`／`--fence`（下記「シーン構造の切り替え」）で条件を統制 |
| `scene_stats.py` | GT深度EXRとGT細線マスクからシーンの見え方を定量（空占有率・**細線背後の空率**・細線画素占有率）。シーン間で背景の性質を揃える/変えるときの検算に使う。要 OpenEXR |
| `analyze_sfm.py` | COLMAP点群をGT座標にUmeyama位置合わせし、細線上のSfM点の欠乏を定量。厳密版=電線スパンの**表面**から3cm以内（**中心線ポリラインへの線分距離**−線半径で判定。半径はgt/scene_params.jsonから自動取得）。欠乏倍率は**電線スパンのみで分子分母を揃えた**主指標と旧互換の混合指標を両方出す（§9ah）。**点が0個のときは倍率を下界（≧N倍）として報告する** |
| `eval_wires.py` | 学習済み3DGSのテストレンダを 全体/細線領域/背景 PSNR に分解して評価し、比較図を出力。加えて2D線検出（Canny→HoughLinesP）ベースの**消失率**（GT細線recallに対するレンダrecallの正規化欠損）を算出し診断図を出力 |
| `inject_points.py` | オラクル点注入（キラー実験）：GTの電線中心線点をUmeyama逆変換でCOLMAP点群に追加し `<dst>/sparse/0` を書き出す。dstにはinput/gt/imagesを用意しておく |
| `run_sweep.py` | 上記＋COLMAP＋学習＋レンダを条件グリッドで一括実行するオーケストレータ。結果は `sweep_results.csv` に1行/条件で追記 |
| `sweep_results.csv` | スイープ結果の集計表（実験名・条件・SfM点数・PSNR分解・落差dB） |
| `detect_recall.csv` | 12実験ぶんの消失率メトリクス集計表（実験名・条件・R_gt・R_ren・正規化recall・消失率） |
| `f40_2x2_runs.csv` | 疎視点（f40 open）2×2実験の全8run集計（素/Mip × 素init/オラクル注入 × 各2run。疎視点域は乱数分散が大きいため全セル2run） |
| `collect_mip_sweep.py` | Mipスイープ（第1波8条件×{素3DGS,Mip}＋アンカー条件の分散2run）の結果を実験dirから集計して `mip_sweep_results.csv` に出力 |
| `plot_mip_sweep.py` | `mip_sweep_results.csv` から3面図 `figs/mip_sweep_radius.png` を生成（第1波と同配色、手法は線種で区別、×印=アンカー再学習run） |
| `mip_sweep_results.csv` | Mipスイープ集計表（実験×手法×run の18行。細線PSNR・gap・消失率・最悪視点recall） |
| `plot_resolution.py` | `resolution_results.csv` から解像度統制の2面図 `figs/resolution_ladder.png` を生成（左＝各解像度内での寄与＝主張、右＝絶対値の梯子） |
| `resolution_results.csv` | 解像度統制（見かけ画素幅 3.0px vs 1.5px）の全run表。**電線のみ／棒／混合／背景／落差／消失率を併記**（§9aj） |
| `analyze_gaussians.py` | 学習済みガウシアン(ply)の解剖：電線スパン近傍の数・不透明度・スケール分布（生値は同名.npzにも保存）＋霧診断。`--fog-dump N --fog-view frame_XXXX` で指定視点の視線を塞ぐガウシアンの素性（色・スパン距離等）をダンプ。**霧の有無の検出はレンダ側指標（R_ren等）で行い、本スクリプトは組成分析に使う**（中心視線ベースのfog_alphaは正当な近傍ジオメトリも拾うため検出用途には不適） |
| `plot_opacity_hist.py` | 電線スパン3cm内ガウシアンの不透明度分布図 `figs/wire_opacity_hist.png` を生成（「素3DGSは透明のまま／Mipで二峰化」の機構図） |
| `ablate_wire_gaussians.py` | 残差分解の介入実験：学習済みplyの電線上ガウシアンを編集（半透明分の削除/不透明化）して再レンダ用モデルdirを作る |
| `measure_wire_width.py` | レンダ上の電線の見かけの太さ（プロファイルFWHM）とディップ深さを測定し、GT比の太り倍率と線中心のサブピクセル位置ずれ分布を出す |
| `error_budget.py` | 残差の誤差収支：細線マスク内の二乗誤差を 輝度/色・位置ずれ/形状・裾集中度 に分解する |
| `tail_map.py` | 裾の空間分布診断：誤差上位断面がどの視点・線上のどこに集中するかを可視化（視点別シェア・隣接クラスタリング率・GT重畳ヒートマップ）。出力は `$EXP/gt/tail_map_<model>_<iter>.{json,png}` |
| `wbgeom.py` | 解析スクリプト共通の幾何ユーティリティ：COLMAP→GT のロバスト位置合わせ（RMSE>2cmなら外れカメラを除いて再推定し一覧を返す）、電線スパンの中心線への線分距離、Mip の 3Dフィルタ込み実効不透明度・実効スケール（2026-09-24 監査） |
| `collect_resolution.py` | 解像度統制の全runを summary から集めて `resolution_results.csv` を作る。PSNRマスク膨張2px（md2）と膨張なし（md0）を併記（md0 が無ければ eval_wires を実行） |
| `collect_disappearance.py` | 消失率を「床」（細線を inpaint で消したGTでも検出される割合＝測れる上限）付きで全条件集計し `disappearance_results.csv` を作る |
| `collect_method_bench.py` | 手法横断ベンチ（アンカー r15 open f80・30k・各3run以上）を集めて `method_bench_results.csv`（1run1行）・`method_bench_summary.csv`（手法ごとの平均・sd、素3DGS対比の Welch p と Holm 補正 p）を作る。手法間の主要対比（LOD-GS−Mip 等）も出す |
| `inject_interaction_stats.py` | オラクル注入×AA の交互作用（7k・電線のみ）を Welch–Satterthwaite t で検定。素側は12runプールと注入と同一設定（newdef 3run）の両方。§9ao の最小検出効果（非心t）も出す。→ `inject_interaction_stats.json` |
| `make_probe_scene.py` / `eval_probe.py` / `collect_probe.py` | 「帯 vs 半平面」検定（設計 memo/design_20260922_band_vs_halfplane.md、研究メモ §9as）。発光円柱（半径6段×傾き3）と板を一様な空に置いたシーンの生成、境界1本あたりの誤差密度の評価（GTぼかし基準も同じ指標で測る）、超過比 X＝(円柱/板)_モデル÷(円柱/板)_ぼかし(σ_fit) の集計と図（`figs/probe_<実験dir名>_<split>.png`）。第2回は `make_probe_scene.py --layout random`（物体数増・長さ統一）。評価・集計とも `--split train` で学習視点（テスト視点は空色フローターが物体を覆うため。§9as） |
| `probe_bootstrap.py` | 同検定の不確かさ：視点ブートストラップの X の95%区間、向き別 R、物体単位の X を太い側（≥60mm）の帰無分布と比較（`--exclude` で消失物体を除く） |
| `probe_presence.py` | 同検定の消失判定：物体マスク内のコントラスト保持率（0≈消失）。境界誤差の超過と「丸ごと消えた」を分ける |
| `probe_sfm.py` | 同検定：物体ごとの SfM 点数（表面から3cm以内）と消失の対応。消失が初期化で説明できるかを見る（§9as-3） |
| `probe_gaussians.py` | 同検定：消えた物体の周り（表面から3cm）のガウシアン数と実効不透明度。描画上の消失が地図からの欠落かを見る（§9as-4） |
| `probe_collision.py` | 同検定を Splat-Nav（arXiv 2403.02751）の衝突幾何（1σ楕円体の和集合）で評価：物体ごとの被覆率・最大の隙間（見逃し）と、空の対照位置の被覆（誤検知）（§9as-5） |
| `map_eval.py` / `collect_map_eval.py` | 3DGS を地図として評価（研究メモ §9at）：Splat-Nav の衝突幾何（1σ楕円体の和集合）で、電線の被覆率・隙間（見逃し）と、GT 深度で空きと確かめた点の誤占有（誤検知、機体半径 κ 付きも）を測る。集計は各 run の電線 PSNR と並べて `map_eval_results.csv` に |
| `splatnav_wires.py` / `collect_splatnav.py` | Splat-Nav の公開実装（Splat-Plan、`tools/splatnav` に clone・`tools/splatnav/venv` で実行・CPU 可）に学習済み地図を渡し、電線をまたぐ経路を計画（研究メモ §9au）。成功率（A* が通れるか）・電線への衝突・経路長の GT 地図比を、不透明度の足切りごとに出す | `--voxel ellip` で A* の格子を楕円体そのものに（§9av）
| `astar_free.py` | Splat-Plan の第1段（ボクセル格子の A*）だけを、Splat-Nav の格子（外接箱）と楕円体の格子の2通りで回す（§9av）。`--mode real` は実データの撮影カメラ位置どうし（歩いた軌道があるので必ず通れるはず）、`--mode synth` は電線をまたぐ組 |
| `sky_carve.py` / `collect_sky_carve.py` | 空の画素の視線上は空いていることを使い、正解の3D形状なしで上空（電線・空の高さ）に既知の空き点を作って、地図の誤占有と、空き点だけでつながる2点の A* を測る（研究メモ §9aw）。空のマスクは色＋SfM 特徴点の除外＋上端連結。合成モードでは空き点の正しさを電線の距離で確かめる | 電線（合成は GT、実データは `wire3d.py` の復元）があれば、地図の電線被覆と経路の電線セル通過も出す
| `wire3d.py` | 実データの電線を多視点のボクセル投票で3D復元（明るい背景の暗い細線の検出 → 粗い/細かい格子で賛成・反対の票 → 芯線抽出）。`--views even` で偶数視点だけで復元し奇数視点で一致を検証（研究メモ §9ay） |
| `sky_prune.py` | 後処理で空のガウシアンを除く（空間彫刻：中心が安全な空の画素に写る視点が `--min-votes` 以上なら除く）。`<model>_skyprune/` に新しい ply を書く。実データは `--views even` で除去し、評価は `sky_carve.py --views odd`（研究メモ §9ax） |
| `phase2/` | Phase 2（AA前提の細線特化最適化）のコード。`train_thin.py`（mip-splatting改の学習: L1画素重み3方式＋視点適応サンプリング。tools/mip-splatting へコピーして venv 実行）・`tail_corr3d.py`（裾の3D対応診断。要 OpenEXR）・`collect_wave1.py`＋`phase2_wave1_results.csv`（第1波の集計） |
| `real/` | **実データ（GTマスク・深度が無い撮影素材）の評価**。GT画像から「青空上の暗い細線」を検出して細線マスクの代用にする（`wire_struct_masks.py`・`wire_extrap.py`＝空背景/構造背景の区間分け）。`eval_real_wires*.py`（細線PSNR・保持率）、`eval_real_open_vs_struct.py`（空背景/構造背景）、`eval_prune_sweep.py`・`analyze_np05.py`（近カメラprune）、`eval_real_lodgs.py`・`lodgs_lod_ablation.py`（LOD-GS）、`hf_calib.py`（高周波保持率の定義）、`real_wire_sfm.py`（SfM欠乏の実データ版）、`real_paired_stats.py`（手法差の seed Welch＋視点単位対検定。`--compare A-B` で基準群以外の対比、領域ごとに Holm 補正）。**マスクは電線の見えやすい区間だけを拾う選択バイアスがある**（監査R-02）。2026-09-24 に tmp/ から移設。`real_splatnav.py`（カメラの地面からの高さで尺度を決める `camera_frame`、Splat-Plan をカメラ位置どうしで実行。遅いので通常は `astar_free.py --mode real` を使う） |
| `patches/` | `tools/` の他手法（git管理外）に当てた修正の差分。**比較実験を再現するには当てる必要がある**。`lodgs_2d2512a.patch`（LOD-GS：単一スケール対応＋チェックポイントに `_coefs` を含める。§9af）・`raings_1bee777.patch`（RAIN-GS：`args_dict` が None のときのガード。§9k）。各ファイル冒頭に元コミットと当て方を記載 |
| `figs/` | スイープ結果・消失率の図 |

## シーン構造の切り替え（複数シーン検証）

結論のシーン非依存性を確かめるため、`make_scene.py` / `run_sweep.py` に直交する2つのフラグがある。
**既定値は従来シーンと完全に同一**で、既定のままなら既存実験と1バイトも変わらない。

| フラグ | 値 | 内容 |
|---|---|---|
| `--layout` | `base`（既定） | 電柱2本・平行な電線3本（従来シーン） |
| | `crossing` | 電柱4本（高さ5.6/3.6m）・2方向の電線6本が画像上で交差する。**構造**を変える軸（線方向の多様性・相互オクルージョン・高さ段違い。背景と線半径は base と同じ） |
| | `street` | **ジオメトリは base と完全同一**のまま、カメラ軌道 r=8.5 の外側（r=11〜14m）に高さ6〜10mの建物リングを置いて電線の背後を高周波の壁面で埋める。**背景**を変える軸（細線背後の空率 52%→6%）。空自体は上方に残るので `--enclosed` とは別物 |
| `--fence` | `periodic`（既定） / `random` | 柵の縦棒を等間隔／ランダム間隔にする。SfM点欠乏が周期構造のマッチング曖昧性由来かを単独で検定するための軸（`--fence-seed` で再現） |

- 実験名は既定から外れた軸だけサフィックスが付く（`_Lcrossing` / `_Frandom`）。
- 電柱の本数・位置は `gt/scene_params.json` に記録され `analyze_sfm.py` が読む（ハードコードしない）。
- `gt/scene_params.json` には画面上の電線幅[px]の分布（`wire_width_px`）も記録される。
  レイアウトで被写体距離が変わるので、シーン比較のときは幅が揃っているかを必ず確認する。

```bash
# crossing シーンで密(f80)・疎(f40)を一括生成→SfM→素3DGS 7k
python3 research/wirebench/run_sweep.py --radii 0.015 --frames 80 40 \
    --variants open --iters 7000 --layout crossing
# 背景の性質の検算
python3 research/wirebench/scene_stats.py experiments/*_f80_open experiments/*_f80_open_Lstreet
```

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

# 5. テスト視点レンダ → 細線評価（PSNR分解＋消失率。位置引数は既定値のままでよい）
python3 /workspace/scripts/run_render.py -m $EXP/output -s $EXP --iteration 7000 --skip_train
cd $EXP && python3 /workspace/research/wirebench/eval_wires.py . 7000
# 30k版を別フォルダで学習した場合: eval_wires.py . 30000 output_30k
# 消失率メトリクスのパラメータを変える場合（既定: tau=2px, band-radius=2px,
# canny-lo=10, canny-hi=30, hough-thresh=8, min-len=8, max-gap=25）:
#   eval_wires.py . 7000 output --tau 3 --band-radius 3
# PSNR用マスクの膨張半径を変える場合（マスク希釈の分析用。既定2=従来どおり。
# 非標準値の結果は wire_eval_*_mdN/ に出て標準評価を上書きしない）:
#   eval_wires.py . 7000 output --mask-dilate 0
```

## 出力（GT一式は `$EXP/gt/`）

- `poses.json` … GTカメラ（cam2world 4x4、Blender/OpenGL規約: 視線-Z・上+Y）＋内部パラメータ
- `mask_thin/NNNN.png` … 細線マスク（白=電線+フェンス棒。pass_index=1のIDMask）
- `depth/NNNN.exr` … 深度（float）
- `wire_points.npz` … 細線の3D点（points: (N,3) world座標 / labels: 1=電線 2=フェンス棒）
- `scene_params.json` … シーン生成条件（wire_radius・frames・enclosed等。分析側が参照する）
- `sfm_analysis.json` … analyze_sfm.py の集計値（run_sweep.pyが読む）
- `wire_eval_*/summary.json` … eval_wires.py の集計値（PSNR分解＋消失率。run_sweep.pyが読む）
- `sfm_points_overlay.png` … SfM点群を視点1に投影した可視化（analyze_sfm.pyが生成）
- `wire_eval_*/` … 比較図（GT|render、拡大クロップ）＋消失率診断図
  `detect_overlay_*.png`（緑=GT細線マスク、赤=レンダの検出線、黄=被覆〈tau px以内で一致〉。
  最も消失率が高い視点で出力。いずれもeval_wires.pyが生成）

## 実装上の注意（ハマりどころ）

- **カラーマネジメント**: AgX(デフォルト)は彩度が潰れSfMに不利、Standardは白飛びする → Filmicを使用
- **テクスチャ**: Brickテクスチャは垂直壁で縞になる → ノイズ系を使用。テクスチャ座標はObject（ワールド）出力
- **--enclosed**: 円筒プリミティブはデフォルトで蓋付き → 底面が地面とZファイトして真っ黒になる
  → `end_fill_type="NOTHING"` にしてある（天井は別の円盤）
- **電柱座標**: analyze_sfm.py の `POLE_XY` は make_scene.py の `pole_positions` と一致させること
- **テスト分割**: gaussian-splatting の `--eval` は名前順の8枚おき（llffhold=8）。
  eval_wires.py はこの規約でレンダ番号→フレーム名を対応させている
- **消失率メトリクスの既知の限界**: `20260705_wb_r15mm_f20_open` はR_gt=0.71と他11実験（すべて≥0.98）より
  明確に低い。原因はテスト視点3枚中1枚（frame_0017、マスク画素の94%を占める）でフェンス棒がボックス背後の
  低コントラスト領域と重なり、GT画像上でも検出器が拾えないため（Canny/Houghのパラメータ調整では解決しない。
  band-radius・tauを広げても改善は僅か）。正規化recall（DR=R_ren/R_gt）はこの実験自身のR_gtを分母に取るため
  実験内の比較は成立するが、他実験とのR_gt絶対値の比較には注意

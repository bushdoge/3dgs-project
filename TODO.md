# TODO

## 実装予定

- [ ] ② COLMAP品質確認 — sparse/0/ の cameras.txt・points3D.txt から登録カメラ数・3D点数・再投影誤差を結果ページに表示する
- [ ] ③ 実験管理の充実 — 06_experiment_manager.py に各ステップのログ閲覧タブと config 設定の表示を追加する
- [ ] ④ パイプライン設定プリセット保存 — よく使う設定（特徴量・FPS・iter数など）を名前をつけて保存・呼び出しできるようにする
- [ ] ⑤ 複数実験のLoss/PSNR重ね比較 — 05_compare.py に複数実験の学習曲線を重ねて表示する機能を追加する

## 完了済み

- [x] ① レンダリング実行UI — 04_results.py にレンダリング実行セクションを追加、scripts/run_render.py 新規作成

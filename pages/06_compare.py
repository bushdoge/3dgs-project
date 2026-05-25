# 複数の実験結果を横並びで比較するページ
# フレーム数・COLMAP状態・PSNR・L1 Loss の学習曲線を重ね表示し、レンダリング画像も並べる

import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.title("⚖️ 実験比較")
st.caption("複数の実験結果を並べて学習曲線・PSNR・レンダリング画像を比較します")

st.divider()

# ── 実験フォルダの列挙 ─────────────────────────────────────────────────────────
experiments_dir = Path("/workspace/experiments")
all_exps = []
if experiments_dir.exists():
    all_exps = sorted(
        [p for p in experiments_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )

if not all_exps:
    st.warning("experiments/ フォルダに実験が見つかりません。")
    st.stop()

exp_names = [p.name for p in all_exps]
selected_names = st.multiselect(
    "比較する実験を選択（複数可）",
    exp_names,
    default=exp_names[:2] if len(exp_names) >= 2 else exp_names,
    help="2〜4件程度を選ぶと見やすいです",
)

if not selected_names:
    st.info("実験を選択してください。")
    st.stop()

selected_exps = [experiments_dir / name for name in selected_names]

# ── カラーパレット割り当て ────────────────────────────────────────────────────
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#636363", "#bcbd22", "#17becf",
]
MARKERS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

exp_colors  = {n: PALETTE[i % len(PALETTE)]  for i, n in enumerate(selected_names)}
exp_markers = {n: MARKERS[i % len(MARKERS)]  for i, n in enumerate(selected_names)}

# 色付き凡例をmultiselect直下に表示
_legend_items = "".join(
    f'<span style="display:inline-flex;align-items:center;gap:5px;'
    f'margin:3px 16px 3px 0;white-space:nowrap;">'
    f'<span style="background:{exp_colors[n]};color:#fff;font-size:0.72rem;'
    f'font-weight:bold;padding:1px 6px;border-radius:3px;">{exp_markers[n]}</span>'
    f'<span style="font-size:0.75rem;color:#b0c8e0;">'
    f'{n if len(n)<=36 else n[:34]+"…"}</span></span>'
    for n in selected_names
)
st.markdown(
    f'<div style="display:flex;flex-wrap:wrap;padding:4px 0 8px;">{_legend_items}</div>',
    unsafe_allow_html=True,
)

st.divider()


# ── ユーティリティ ─────────────────────────────────────────────────────────────
def count_images(folder):
    p = Path(folder)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".png", ".jpeg"})


def get_training_records(exp_path):
    """学習ログから PSNR・L1・SSIM・LPIPS を取得する。
    返す dict: {"iteration": int, "split": str, "PSNR": float, "L1": float, ...}
    """
    records = []
    log_candidates = (
        list((exp_path / "output").rglob("train_log.txt")) +
        list((exp_path / "output").rglob("*.txt"))
    ) if (exp_path / "output").exists() else []

    for log_file in log_candidates:
        text = log_file.read_text(errors="replace")
        for m in re.finditer(
            r"\[ITER (\d+)\] Evaluating (\w+): L1 ([^\s]+) PSNR ([^\s]+)"
            r"(?:\s+SSIM ([^\s]+))?(?:\s+LPIPS ([^\s\[]+))?",
            text,
        ):
            try:
                rec = {
                    "iteration": int(m.group(1)),
                    "split":     m.group(2),
                    "L1":        float(m.group(3)),
                    "PSNR":      float(m.group(4)),
                }
                if m.group(5):
                    rec["SSIM"]  = float(m.group(5))
                if m.group(6):
                    rec["LPIPS"] = float(m.group(6))
                records.append(rec)
            except ValueError:
                pass
        if records:
            break
    return records


def get_actual_resolution(exp_path):
    """input/ の画像1枚からWxHを取得する。取得できなければNone"""
    from PIL import Image as _PIL
    for d in ["input", "images"]:
        imgs = list((exp_path / d).glob("*.jpg")) + list((exp_path / d).glob("*.png"))
        if imgs:
            try:
                w, h = _PIL.open(imgs[0]).size
                return w, h
            except Exception:
                pass
    return None


def get_colmap_stats(exp_path):
    """sparse/0/images.txt と points3D.txt から COLMAP再構成品質を取得する"""
    sparse_dir = exp_path / "sparse" / "0"
    if not sparse_dir.exists():
        return {}

    n_registered = 0
    images_txt = sparse_dir / "images.txt"
    if images_txt.exists():
        lines = [l for l in images_txt.read_text(errors="replace").splitlines()
                 if l.strip() and not l.startswith("#")]
        n_registered = len(lines) // 2  # 2行で1画像（姿勢行 + 特徴点行）

    n_points = 0
    mean_reproj = None
    points_txt = sparse_dir / "points3D.txt"
    if points_txt.exists():
        data_lines = [l for l in points_txt.read_text(errors="replace").splitlines()
                      if l.strip() and not l.startswith("#")]
        n_points = len(data_lines)
        errors = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 8:
                try:
                    errors.append(float(parts[7]))
                except ValueError:
                    pass
        if errors:
            mean_reproj = sum(errors) / len(errors)

    return {"n_registered": n_registered, "n_points": n_points, "mean_reproj": mean_reproj}


def get_renders(exp_path):
    images = []
    for rd in [exp_path / "renders"] + list(
        (exp_path / "output").rglob("renders") if (exp_path / "output").exists() else []
    ):
        if rd.is_dir():
            images += sorted(rd.rglob("*.png")) + sorted(rd.rglob("*.jpg"))
    return images


# ── 全実験のレコード収集 ──────────────────────────────────────────────────────
all_records: dict[str, list] = {}
for exp in selected_exps:
    all_records[exp.name] = get_training_records(exp)

# ── サマリーテーブル ──────────────────────────────────────────────────────────
st.subheader("📊 実験サマリー")

rows = []
for exp in selected_exps:
    n_input   = count_images(exp / "input")
    has_colmap = (exp / "sparse" / "0").exists()
    has_output = (exp / "output").exists()
    ply_files  = list((exp / "output").rglob("*.ply")) if has_output else []
    records    = all_records[exp.name]

    best_psnr_test  = best_psnr_train  = None
    best_ssim_test  = best_ssim_train  = None
    min_lpips_test  = min_lpips_train  = None
    min_l1_test     = min_l1_train     = None
    if records:
        test_r  = [r for r in records if r["split"] == "test"]
        train_r = [r for r in records if r["split"] == "train"]
        for split_r, suffix in [(test_r, "test"), (train_r, "train")]:
            if not split_r:
                continue
            psnr_vals  = [r["PSNR"]  for r in split_r]
            l1_vals    = [r["L1"]    for r in split_r]
            ssim_vals  = [r["SSIM"]  for r in split_r if "SSIM"  in r]
            lpips_vals = [r["LPIPS"] for r in split_r if "LPIPS" in r]
            if suffix == "test":
                best_psnr_test  = max(psnr_vals)
                min_l1_test     = min(l1_vals)
                if ssim_vals:  best_ssim_test  = max(ssim_vals)
                if lpips_vals: min_lpips_test  = min(lpips_vals)
            else:
                best_psnr_train = max(psnr_vals)
                min_l1_train    = min(l1_vals)
                if ssim_vals:  best_ssim_train = max(ssim_vals)
                if lpips_vals: min_lpips_train = min(lpips_vals)

    note_path = exp / "note.md"
    note_preview = ""
    if note_path.exists():
        lines = note_path.read_text(encoding="utf-8").strip().splitlines()
        note_preview = lines[0][:40] if lines else ""

    rows.append({
        " ":                  exp_markers[exp.name],  # カラーマーカー列
        "実験名":             exp.name,
        "フレーム数":         n_input,
        "COLMAP":             "✅" if has_colmap else "❌",
        "学習":               f"✅ ({len(ply_files)}ply)" if ply_files else ("⏳" if has_output else "❌"),
        "PSNR test (dB)":     f"{best_psnr_test:.2f}"   if best_psnr_test  is not None else "-",
        "PSNR train (dB)":    f"{best_psnr_train:.2f}"  if best_psnr_train is not None else "-",
        "SSIM test":          f"{best_ssim_test:.4f}"   if best_ssim_test  is not None else "-",
        "SSIM train":         f"{best_ssim_train:.4f}"  if best_ssim_train is not None else "-",
        "LPIPS test":         f"{min_lpips_test:.4f}"   if min_lpips_test  is not None else "-",
        "LPIPS train":        f"{min_lpips_train:.4f}"  if min_lpips_train is not None else "-",
        "L1 test":            f"{min_l1_test:.4f}"      if min_l1_test     is not None else "-",
        "L1 train":           f"{min_l1_train:.4f}"     if min_l1_train    is not None else "-",
        "メモ":               note_preview,
    })

df_summary = pd.DataFrame(rows)

def _style_summary(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for idx, row in df.iterrows():
        color = exp_colors.get(row["実験名"], "#888")
        styles.loc[idx, " "] = (
            f"background-color:{color};color:#fff;"
            f"font-weight:bold;text-align:center;font-size:0.9rem;"
        )
        for col in df.columns:
            if col != " ":
                styles.loc[idx, col] = f"border-left:3px solid {color}22;"
    return styles

# ── パイプライン設定テーブル（pipeline_config.json が存在する実験のみ） ──────
_cfg_loaded = {}
for exp in selected_exps:
    _p = exp / "pipeline_config.json"
    if _p.exists():
        try:
            _cfg_loaded[exp.name] = json.loads(_p.read_text(encoding="utf-8"))
        except Exception:
            pass

if _cfg_loaded:
    _cfg_rows = []
    for exp in selected_exps:
        _c = _cfg_loaded.get(exp.name, {})
        _hloc    = _c.get("use_hloc")
        _pair    = _c.get("pair_method", "")
        _iters   = _c.get("iterations")
        _res      = _c.get("resolution")
        _act_res  = get_actual_resolution(exp)
        if _res is None:
            if _act_res:
                _res_str = f"{_act_res[0]}×{_act_res[1]} (自動)"
            else:
                _res_str = "自動"
        else:
            if _act_res:
                _scaled_w = _act_res[0] // _res
                _scaled_h = _act_res[1] // _res
                _res_str = f"{_scaled_w}×{_scaled_h} (1/{_res}x)"
            else:
                _res_str = f"1/{_res}x"

        _n_input = count_images(exp / "input")
        _cs = get_colmap_stats(exp)
        _n_reg   = _cs.get("n_registered", 0)
        _n_pts   = _cs.get("n_points", 0)
        _reproj  = _cs.get("mean_reproj")
        _reg_rate = f"{_n_reg / _n_input * 100:.1f}%" if _n_input > 0 and _n_reg > 0 else "-"

        _cfg_rows.append({
            " ":              exp_markers[exp.name],
            "実験名":         exp.name,
            "姿勢推定":       ("HLoc" if _hloc else "COLMAP") if _hloc is not None else "-",
            "特徴点抽出器":   (_c.get("feature_type", "-") if _hloc
                               else _c.get("camera_model", "-")) if _c else "-",
            "マッチャー":     _c.get("matcher_type", "-") if (_c and _hloc) else "-",
            "top-K":          str(_c.get("num_matched", "-")) if (_c and _hloc and _pair == "retrieval") else "-",
            "登録カメラ数":   str(_n_reg) if _n_reg else "-",
            "登録率":         _reg_rate,
            "3D点群数":       f"{_n_pts:,}" if _n_pts else "-",
            "再投影誤差(px)": f"{_reproj:.4f}" if _reproj is not None else "-",
            "学習ステップ数": str(_iters) if _iters is not None else "-",
            "eval":           str(_c.get("eval", "-")) if _c else "-",
            "解像度":         _res_str if _c else "-",
        })
    _df_cfg = pd.DataFrame(_cfg_rows)
    st.dataframe(
        _df_cfg.style.apply(_style_summary, axis=None),
        use_container_width=True,
        hide_index=True,
        column_config={
            " ":              st.column_config.TextColumn(width="small"),
            "姿勢推定":       st.column_config.TextColumn(help="カメラ姿勢推定に使うエンジン。HLoc（深層学習ベース）または生COLMAP（SIFT）。"),
            "特徴点抽出器":   st.column_config.TextColumn(help="画像から特徴点を検出するアルゴリズム。SuperPoint・ALIKED は深層学習ベースで頑健。"),
            "マッチャー":     st.column_config.TextColumn(help="2枚の画像間で特徴点を対応付けるアルゴリズム。LightGlue は Transformer ベースの高精度マッチャー。"),
            "top-K":          st.column_config.TextColumn(help="retrieval マッチング時に各画像と照合する類似画像の上位K枚数。多いほど漏れが減るが時間がかかる。"),
            "登録カメラ数":   st.column_config.TextColumn(help="COLMAP が3D再構成に取り込めた（姿勢を推定できた）画像枚数。"),
            "登録率":         st.column_config.TextColumn(help="入力フレーム総数に対して姿勢推定に成功した割合。高いほど再構成が密になる。"),
            "3D点群数":       st.column_config.TextColumn(help="再構成された3次元点の総数。多いほど密な点群が得られているが、多すぎる場合はノイズも含む可能性あり。"),
            "再投影誤差(px)": st.column_config.TextColumn(help="3D点を画像に投影した際の予測座標と実測座標のズレの平均（ピクセル単位）。小さいほどカメラ姿勢推定の精度が高い。目安：1px以下が良好。"),
            "学習ステップ数": st.column_config.TextColumn(help="3DGS学習のイテレーション数。多いほど精細になるが時間もかかる。"),
            "eval":           st.column_config.TextColumn(help="学習時にテスト分割で評価指標（PSNR等）を計算するか。True にすると学習ログにスコアが記録される。"),
            "解像度":         st.column_config.TextColumn(help="学習に使用した画像の解像度。「自動」は入力画像そのまま（1600px超なら自動縮小）。"),
        },
    )
    if len(_cfg_loaded) < len(selected_exps):
        _missing = [n for n in selected_names if n not in _cfg_loaded]
        st.caption(f"⚠️ pipeline_config.json なし（旧形式）: {', '.join(_missing)}")
else:
    st.info("選択中の実験に pipeline_config.json が見つかりません。次回のパイプライン実行から自動で保存されます。")

st.caption("▼ 学習スコアサマリー")

st.dataframe(
    df_summary.style.apply(_style_summary, axis=None),
    use_container_width=True,
    hide_index=True,
    column_config={
        " ":               st.column_config.TextColumn(width="small"),
        "フレーム数":      st.column_config.TextColumn(help="input/ フォルダの画像枚数。COLMAPへの入力フレーム総数。"),
        "COLMAP":          st.column_config.TextColumn(help="COLMAP再構成が完了しているか（sparse/0/ が存在するか）。"),
        "学習":            st.column_config.TextColumn(help="3DGS学習の完了状況と出力された .ply ファイル数。"),
        "PSNR test (dB)":  st.column_config.TextColumn(help="テスト画像での最高 PSNR（Peak Signal-to-Noise Ratio）。高いほど高品質。目安：25dB以上が良好、30dB以上が優秀。"),
        "PSNR train (dB)": st.column_config.TextColumn(help="学習画像での最高 PSNR。test より大幅に高い場合は過学習の可能性あり。"),
        "SSIM test":       st.column_config.TextColumn(help="テスト画像での最高 SSIM（Structural Similarity Index）。輝度・コントラスト・構造の類似度。0〜1で高いほど良い。"),
        "SSIM train":      st.column_config.TextColumn(help="学習画像での最高 SSIM。"),
        "LPIPS test":      st.column_config.TextColumn(help="テスト画像での最小 LPIPS（Learned Perceptual Image Patch Similarity）。人間の知覚に基づく距離。低いほど良い。"),
        "LPIPS train":     st.column_config.TextColumn(help="学習画像での最小 LPIPS。"),
        "L1 test":         st.column_config.TextColumn(help="テスト画像での最小 L1 損失（ピクセル値の平均絶対誤差）。低いほど良い。"),
        "L1 train":        st.column_config.TextColumn(help="学習画像での最小 L1 損失。"),
        "メモ":            st.column_config.TextColumn(help="note.md の1行目のプレビュー。"),
    },
)

# ── 学習曲線の重ね比較 ────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 学習曲線の比較")
st.caption("各実験の学習ログから指標を抽出して重ね表示します。")

any_records = any(all_records[e.name] for e in selected_exps)

if not any_records:
    st.info("学習ログにデータが見つかりません。3DGS学習を完了してください。")
else:
    # 利用可能な指標を全レコードから収集
    all_metric_keys = set()
    for recs in all_records.values():
        for r in recs:
            all_metric_keys.update(r.keys() - {"iteration", "split"})
    METRIC_OPTIONS = [m for m in ["PSNR", "SSIM", "LPIPS", "L1"] if m in all_metric_keys]
    METRIC_HIGHER_IS_BETTER = {"PSNR": True, "SSIM": True, "LPIPS": False, "L1": False}

    fc1, fc2 = st.columns([2, 3])
    with fc1:
        split_filter = st.radio(
            "表示する split", ["test", "train", "両方"],
            horizontal=True, index=0,
        )
    with fc2:
        metric_label = st.radio(
            "指標", METRIC_OPTIONS,
            horizontal=True, index=0,
        )
    metric_col = metric_label

    splits = ["test", "train"] if split_filter == "両方" else [split_filter]

    chart_series: dict[str, dict[int, float]] = {}
    for exp in selected_exps:
        records = all_records[exp.name]
        for sp in splits:
            filtered = [r for r in records if r["split"] == sp and metric_col in r]
            if not filtered:
                continue
            marker = exp_markers[exp.name]
            label  = (f"{marker}" if split_filter != "両方"
                      else f"{marker} {sp}")
            chart_series[label] = {r["iteration"]: r[metric_col] for r in filtered}

    # 系列名 → 実験色のマッピング
    def _series_color(label):
        for exp_name in selected_names:
            m = exp_markers[exp_name]
            if label == m or label.startswith(m + " "):
                return exp_colors[exp_name]
        return "#888888"

    if chart_series:
        series_names  = list(chart_series.keys())
        series_colors = [_series_color(s) for s in series_names]
        color_scale   = alt.Scale(domain=series_names, range=series_colors)

        all_iters = sorted(set(it for vals in chart_series.values() for it in vals))
        rows_long = [
            {"iteration": it, "系列": name, metric_col: vals.get(it)}
            for name, vals in chart_series.items()
            for it in all_iters
            if vals.get(it) is not None
        ]
        chart_df_long = pd.DataFrame(rows_long)
        chart = (
            alt.Chart(chart_df_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("iteration:Q", title="Iteration",
                         axis=alt.Axis(grid=True)),
                y=alt.Y(f"{metric_col}:Q", title=metric_col,
                         axis=alt.Axis(grid=True)),
                color=alt.Color("系列:N",
                                scale=color_scale,
                                legend=alt.Legend(title="")),
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("**最良値サマリー**")
        higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric_col, True)
        marker_to_name = {v: k for k, v in exp_markers.items()}
        stat_rows = []
        for label, vals in chart_series.items():
            if not vals:
                continue
            best_iter = (max if higher_is_better else min)(vals, key=vals.get)
            best_val  = vals[best_iter]
            marker = label.split()[0]
            stat_rows.append({
                "系列": label,
                "実験名": marker_to_name.get(marker, ""),
                f"{'最高' if higher_is_better else '最小'} {metric_col}": f"{best_val:.4f}",
                "達成 iter": best_iter,
            })
        st.dataframe(pd.DataFrame(stat_rows).set_index("系列"), use_container_width=True)
    else:
        st.info("選択した条件に合うデータがありません。")

# ── レンダリング画像比較 ──────────────────────────────────────────────────────
st.divider()
st.subheader("🖼️ レンダリング画像比較")

exp_renders = {exp.name: get_renders(exp) for exp in selected_exps}
has_any_renders = any(imgs for imgs in exp_renders.values())

if not has_any_renders:
    st.info("レンダリング画像が見つかりません。render.py を実行すると生成されます。")
else:
    max_imgs = max(len(imgs) for imgs in exp_renders.values() if imgs)
    n_show = st.slider("表示するコマ数", 1, min(max_imgs, 10), min(3, max_imgs))

    for exp in selected_exps:
        imgs = exp_renders.get(exp.name, [])
        if not imgs:
            continue
        st.markdown(f"**{exp.name}**")
        cols = st.columns(n_show)
        step = max(1, len(imgs) // n_show)
        display = [imgs[i * step] for i in range(n_show) if i * step < len(imgs)]
        for col, img in zip(cols, display):
            col.image(str(img), caption=img.name, use_container_width=True)

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### 実験比較の使い方

1. 上部のマルチセレクトで比較したい実験を選択します（複数選択可）
2. **学習曲線タブ**：PSNRとL1 Lossの推移を重ね書きで比較できます
3. **レンダリング比較タブ**：同一シーンで学習した複数モデルのレンダリング結果を並べて確認できます

---

### グラフの見方

| グラフ | 良い状態 |
|---|---|
| **PSNR** | 上に向かうほど良い。高いほど高品質 |
| **L1 Loss** | 下に向かうほど良い。低いほど精度高 |

- **train PSNR**：学習画像からの評価（過学習の参考）
- **test PSNR**：未学習画像からの評価（`--eval` 有効時のみ表示）

---

### 比較のポイント

- 同じシーンで異なるパラメータ（学習ステップ数・FPS・eval有無）を試すと有効です
- ステップ数が多い実験ほど学習曲線が長く表示されます
- PSNRが途中で飽和している場合はそれ以上学習しても改善が見込めません

---

### レンダリング比較
- 「📷 結果確認」ページでレンダリング実行後に表示されます
- 実験フォルダ内の `output/render/` に画像が保存されています
""")
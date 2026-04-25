# 複数の実験結果を横並びで比較するページ
# フレーム数・COLMAP状態・PSNR・L1 Loss の学習曲線を重ね表示し、レンダリング画像も並べる

import re
from pathlib import Path

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

df_summary = pd.DataFrame(rows).set_index("実験名")
st.dataframe(df_summary, use_container_width=True)

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
            label = exp.name if split_filter != "両方" else f"{exp.name} ({sp})"
            chart_series[label] = {r["iteration"]: r[metric_col] for r in filtered}

    if chart_series:
        all_iters = sorted(set(it for vals in chart_series.values() for it in vals))
        chart_df = pd.DataFrame(
            {name: [vals.get(it) for it in all_iters] for name, vals in chart_series.items()},
            index=all_iters,
        )
        chart_df.index.name = "iteration"
        st.line_chart(chart_df)

        st.markdown("**最良値サマリー**")
        higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric_col, True)
        stat_rows = []
        for label, vals in chart_series.items():
            if not vals:
                continue
            best_iter = (max if higher_is_better else min)(vals, key=vals.get)
            best_val  = vals[best_iter]
            stat_rows.append({
                "系列": label,
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

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass

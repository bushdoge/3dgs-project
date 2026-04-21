# 複数の実験結果を横並びで比較するページ
# フレーム数・COLMAP状態・PSNR・レンダリング画像を一覧・比較表示する

import re
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="実験比較", page_icon="⚖️", layout="wide")

st.title("⚖️ 実験比較")
st.caption("複数の実験結果を並べてPSNRやレンダリング画像を比較します")

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


def get_psnr_records(exp_path):
    """学習ログからPSNR値を取得する"""
    records = []
    log_candidates = (
        list((exp_path / "output").rglob("train_log.txt")) +
        list((exp_path / "output").rglob("*.txt")) +
        list((exp_path / "logs").rglob("*.txt"))
    ) if (exp_path / "output").exists() else []

    for log_file in log_candidates:
        text = log_file.read_text(errors="replace")
        for m in re.finditer(
            r"\[ITER (\d+)\] Evaluating (\w+): L1 [^\s]+ PSNR [^(\n]*\(([\d.]+)\)",
            text,
        ):
            records.append({
                "iteration": int(m.group(1)),
                "split": m.group(2),
                "PSNR": float(m.group(3)),
            })
        if records:
            break
    return records


def get_renders(exp_path):
    """レンダリング画像のパスリストを取得する"""
    images = []
    for rd in [exp_path / "renders"] + list((exp_path / "output").rglob("renders") if (exp_path / "output").exists() else []):
        if rd.is_dir():
            images += sorted(rd.rglob("*.png")) + sorted(rd.rglob("*.jpg"))
    return images


# ── サマリーテーブル ──────────────────────────────────────────────────────────
st.subheader("📊 実験サマリー")

import pandas as pd

rows = []
for exp in selected_exps:
    n_input = count_images(exp / "input")
    has_colmap = (exp / "sparse" / "0").exists()
    has_output = (exp / "output").exists()
    ply_files = list((exp / "output").rglob("*.ply")) if has_output else []
    psnr_records = get_psnr_records(exp)

    best_psnr_test = None
    best_psnr_train = None
    if psnr_records:
        test_vals = [r["PSNR"] for r in psnr_records if r["split"] == "test"]
        train_vals = [r["PSNR"] for r in psnr_records if r["split"] == "train"]
        best_psnr_test = max(test_vals) if test_vals else None
        best_psnr_train = max(train_vals) if train_vals else None

    note_path = exp / "note.md"
    note_preview = ""
    if note_path.exists():
        first_line = note_path.read_text(encoding="utf-8").strip().splitlines()
        note_preview = first_line[0][:40] if first_line else ""

    rows.append({
        "実験名": exp.name,
        "フレーム数": n_input,
        "COLMAP": "✅" if has_colmap else "❌",
        "学習": f"✅ ({len(ply_files)}ply)" if ply_files else ("⏳" if has_output else "❌"),
        "PSNR (test)": f"{best_psnr_test:.2f} dB" if best_psnr_test else "-",
        "PSNR (train)": f"{best_psnr_train:.2f} dB" if best_psnr_train else "-",
        "メモ": note_preview,
    })

df_summary = pd.DataFrame(rows).set_index("実験名")
st.dataframe(df_summary, use_container_width=True)

# ── PSNR 比較グラフ ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 PSNR比較グラフ（評価ステップ別）")

all_psnr = {}
for exp in selected_exps:
    records = get_psnr_records(exp)
    if records:
        for split in ("test", "train"):
            key = f"{exp.name} ({split})"
            vals = {r["iteration"]: r["PSNR"] for r in records if r["split"] == split}
            if vals:
                all_psnr[key] = vals

if all_psnr:
    all_iters = sorted(set(it for vals in all_psnr.values() for it in vals))
    chart_data = pd.DataFrame(
        {name: [vals.get(it) for it in all_iters] for name, vals in all_psnr.items()},
        index=all_iters,
    )
    chart_data.index.name = "iteration"
    st.line_chart(chart_data)
else:
    st.info("学習ログにPSNRデータが見つかりません。")

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

    # 各実験のN枚を並べて表示
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

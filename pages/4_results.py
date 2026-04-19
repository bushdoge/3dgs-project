# 3DGS学習結果（レンダリング画像・ログ・ファイル構造）を確認するページ

import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="結果確認", page_icon="🖼️", layout="wide")

st.title("🖼️ 結果確認")
st.caption("学習結果のレンダリング画像・ログ・ファイル構造を確認します")

st.divider()

# ── 実験フォルダ選択 ──────────────────────────────────────────────────────────
experiments_dir = Path("/workspace/experiments")
experiment_list = []
if experiments_dir.exists():
    experiment_list = sorted([
        str(p) for p in experiments_dir.iterdir() if p.is_dir()
    ], reverse=True)

if not experiment_list:
    st.warning("experiments/ フォルダに実験結果が見つかりません。")
    st.stop()

selected_exp = st.selectbox("実験フォルダを選択", experiment_list,
                             format_func=lambda x: Path(x).name)
exp_path = Path(selected_exp)

st.divider()

# ── フォルダ内の状態サマリー ──────────────────────────────────────────────────
st.subheader("📁 フォルダ構造")

def count_files(folder, exts):
    p = Path(folder)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in exts)

col1, col2, col3, col4 = st.columns(4)
with col1:
    n_frames = count_files(exp_path / "frames", {".jpg", ".png", ".jpeg"})
    st.metric("フレーム枚数", n_frames)
with col2:
    has_colmap = (exp_path / "colmap" / "sparse").exists()
    st.metric("COLMAPの状態", "✅ 完了" if has_colmap else "❌ 未実行")
with col3:
    has_output = (exp_path / "output").exists()
    st.metric("学習出力", "✅ あり" if has_output else "❌ なし")
with col4:
    n_renders = count_files(exp_path / "renders", {".jpg", ".png"})
    st.metric("レンダリング画像", n_renders)

# ── レンダリング画像の表示 ─────────────────────────────────────────────────────
st.divider()
st.subheader("🖼️ レンダリング画像")

renders_dir = exp_path / "renders"
if renders_dir.exists():
    images = sorted(renders_dir.rglob("*.png")) + sorted(renders_dir.rglob("*.jpg"))
    if images:
        cols_per_row = st.slider("1行あたりの表示枚数", 2, 6, 4)
        for i in range(0, min(len(images), 24), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(images):
                    col.image(str(images[i + j]), caption=images[i + j].name, use_container_width=True)
    else:
        st.info("renders/ フォルダに画像が見つかりません。")
else:
    st.info("renders/ フォルダがありません。")

# ── 学習ログの表示 ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 学習ログ")

logs_dir = exp_path / "logs"
log_files = []
if logs_dir.exists():
    log_files = sorted(logs_dir.glob("*.txt")) + sorted(logs_dir.glob("*.log"))

if log_files:
    selected_log = st.selectbox("ログファイルを選択", log_files,
                                 format_func=lambda x: x.name)
    with open(selected_log, "r") as f:
        log_content = f.read()
    st.text_area("ログ内容", log_content, height=300)
else:
    # output フォルダ内のログも探す
    output_logs = list((exp_path / "output").rglob("*.txt")) if (exp_path / "output").exists() else []
    if output_logs:
        selected_log = st.selectbox("ログファイルを選択", output_logs,
                                     format_func=lambda x: x.name)
        with open(selected_log, "r") as f:
            log_content = f.read()
        st.text_area("ログ内容", log_content, height=300)
    else:
        st.info("ログファイルが見つかりません。")

# ── config.yaml の表示 ────────────────────────────────────────────────────────
st.divider()
st.subheader("⚙️ 実験設定（config.yaml）")

config_path = exp_path / "config.yaml"
if config_path.exists():
    with open(config_path, "r") as f:
        st.code(f.read(), language="yaml")
else:
    st.info("config.yaml が見つかりません。")

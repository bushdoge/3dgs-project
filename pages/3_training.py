# 3D Gaussian Splattingの学習をtrain.pyで実行するページ

import streamlit as st
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="3DGS学習実行", page_icon="🧠", layout="wide")

st.title("🧠 3DGS 学習実行")
st.caption("3D Gaussian Splatting の学習を実行します（gaussian-splatting/train.py 使用）")

st.divider()

# ── 入力設定 ──────────────────────────────────────────────────────────────────
st.subheader("入力設定（COLMAPの出力フォルダを指定）")

experiments_dir = Path("/workspace/experiments")
colmap_dirs = []
if experiments_dir.exists():
    colmap_dirs = sorted([
        str(p.parent) for p in experiments_dir.glob("*/colmap/sparse") if p.is_dir()
    ])

if colmap_dirs:
    source_path = st.selectbox("実験フォルダ（colmap/sparse が含まれるもの）", colmap_dirs)
else:
    st.warning("COLMAPの出力が見つかりません。先にCOLMAP実行を完了してください。")
    source_path = st.text_input("実験フォルダのパスを直接入力",
                                placeholder="/workspace/experiments/20240101_120000_scene1")

# ── 学習パラメータ ────────────────────────────────────────────────────────────
st.subheader("学習パラメータ")

col1, col2, col3 = st.columns(3)

with col1:
    iterations = st.number_input("学習ステップ数", min_value=1000, max_value=100000,
                                  value=30000, step=1000,
                                  help="デフォルトは30000。短時間で試したい場合は7000程度でも可。")

with col2:
    save_iterations = st.text_input("保存タイミング（カンマ区切り）", value="7000,30000",
                                    help="指定ステップ数ごとにチェックポイントを保存します")

with col3:
    test_iterations = st.text_input("評価タイミング（カンマ区切り）", value="7000,30000")

# ── 出力設定 ──────────────────────────────────────────────────────────────────
st.subheader("出力設定")

if source_path:
    default_output = str(Path(source_path) / "output")
else:
    default_output = ""

model_path = st.text_input("モデル出力フォルダ", value=default_output)

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")

save_list = [s.strip() for s in save_iterations.split(",") if s.strip()]
test_list = [t.strip() for t in test_iterations.split(",") if t.strip()]

cmd_parts = [
    f"python /workspace/scripts/run_train.py",
    f"--source \"{source_path}\"",
    f"--model_path \"{model_path}\"",
    f"--iterations {iterations}",
]
if save_list:
    cmd_parts.append("--save_iterations " + " ".join(save_list))
if test_list:
    cmd_parts.append("--test_iterations " + " ".join(test_list))

st.code(" \\\n  ".join(cmd_parts), language="bash")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()

st.error("⚠️ 学習はGPUを長時間占有します。他の処理が動いていないか確認してから実行してください。")

confirm = st.checkbox("上記を確認しました。学習を開始します。")

if st.button("▶ 学習を開始", type="primary", disabled=not (source_path and confirm)):
    if not os.path.exists(source_path):
        st.error(f"フォルダが見つかりません: {source_path}")
    else:
        os.makedirs(model_path, exist_ok=True)
        st.info(f"学習を開始しました。ログは `{model_path}` に保存されます。")
        st.info("処理が完了するまでこのページは開いたままにしてください。")

        run_args = ["python", "/workspace/scripts/run_train.py",
                    "--source", source_path,
                    "--model_path", model_path,
                    "--iterations", str(iterations)]
        if save_list:
            run_args += ["--save_iterations"] + save_list
        if test_list:
            run_args += ["--test_iterations"] + test_list

        with st.spinner(f"学習中... （最大 {iterations} ステップ）"):
            result = subprocess.run(run_args, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("学習が完了しました！「結果確認」ページで結果を確認してください。")
        else:
            st.error("学習中にエラーが発生しました。")

        with st.expander("ログを表示"):
            st.text(result.stdout or "（出力なし）")
            if result.stderr:
                st.text("STDERR:\n" + result.stderr)

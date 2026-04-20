# COLMAPを使ってカメラ姿勢推定（Structure from Motion）を実行するページ
# gaussian-splatting/convert.py を経由して実行する。入力は experiment/input/、出力は experiment/sparse/0/

import streamlit as st
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="COLMAP実行", page_icon="📷", layout="wide")

st.title("📷 COLMAP実行")
st.caption("フレーム画像からカメラ姿勢を推定します（Structure from Motion）")

st.divider()

# ── 入力設定 ──────────────────────────────────────────────────────────────────
st.subheader("入力設定")

experiments_dir = Path("/workspace/experiments")
exp_dirs = []
if experiments_dir.exists():
    # input/ フォルダが存在し、かつ画像が入っている実験ディレクトリを列挙
    for p in sorted(experiments_dir.iterdir()):
        if p.is_dir():
            inp = p / "input"
            if inp.exists() and (list(inp.glob("*.jpg")) or list(inp.glob("*.png"))):
                n = len(list(inp.glob("*.jpg"))) + len(list(inp.glob("*.png")))
                exp_dirs.append((str(p), n))

col1, col2 = st.columns(2)

with col1:
    if exp_dirs:
        labels = [f"{Path(d).name}  （{n}枚）" for d, n in exp_dirs]
        idx = st.selectbox("実験フォルダ（input/ フォルダを含むもの）",
                           range(len(exp_dirs)), format_func=lambda i: labels[i])
        source_path = exp_dirs[idx][0]
    else:
        st.warning("experiments/ 配下にフレームが入った input/ フォルダが見つかりません。\n"
                   "先にフレーム抽出を実行してください。")
        source_path = st.text_input("実験フォルダのパスを直接入力",
                                    placeholder="/workspace/experiments/20240101_120000_scene1")

with col2:
    camera_model = st.selectbox(
        "カメラモデル",
        ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
        help="通常はOPENCV推奨。360度変換済みや合成画像はPINHOLE。",
    )

# ── 詳細設定 ──────────────────────────────────────────────────────────────────
with st.expander("詳細設定"):
    use_gpu = st.checkbox("GPU使用（SIFT特徴点抽出）", value=True)

# ── 出力先の説明 ──────────────────────────────────────────────────────────────
if source_path:
    st.info(
        f"**出力先（自動）:**\n"
        f"- `{source_path}/sparse/0/` — カメラ姿勢（疎な点群）\n"
        f"- `{source_path}/images/` — アンディストート済み画像"
    )

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")

cmd_args = [
    "python /workspace/scripts/run_colmap.py",
    f'--source_path "{source_path}"',
    f"--camera_model {camera_model}",
]
if not use_gpu:
    cmd_args.append("--no_gpu")

st.code(" \\\n  ".join(cmd_args), language="bash")

# ── 既存結果の確認 ────────────────────────────────────────────────────────────
if source_path:
    sparse_done = Path(source_path) / "sparse" / "0"
    if sparse_done.exists():
        st.success("✅ すでにCOLMAPの出力（sparse/0/）が存在します。再実行すると上書きされます。")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()
st.warning("⚠️ COLMAPはGPUを使用します。フレーム数によって数分〜数十分かかります。")

if st.button("▶ COLMAPを実行", type="primary", disabled=not source_path):
    input_dir = Path(source_path) / "input"
    if not input_dir.exists():
        st.error(f"input/ フォルダが見つかりません: {input_dir}")
    else:
        st.info("COLMAPを実行中です。しばらくお待ちください...")

        run_args = ["python", "/workspace/scripts/run_colmap.py",
                    "--source_path", source_path,
                    "--camera_model", camera_model]
        if not use_gpu:
            run_args.append("--no_gpu")

        with st.spinner("COLMAP実行中（数分〜数十分かかります）..."):
            result = subprocess.run(run_args, capture_output=True, text=True)

        if result.returncode == 0:
            sparse_dir = Path(source_path) / "sparse" / "0"
            if sparse_dir.exists():
                n_cameras = len(list(sparse_dir.glob("*.bin")))
                st.success("COLMAP が正常に完了しました！")
                st.metric("sparse/0/ のファイル数", n_cameras)
                st.info(f"次のステップ：「🧠 3DGS学習実行」ページで `{source_path}` を選択してください。")
            else:
                st.warning("完了しましたが sparse/0/ が見つかりません。ログを確認してください。")
        else:
            st.error("COLMAPの実行中にエラーが発生しました。")

        with st.expander("ログを表示"):
            st.text(result.stdout or "（出力なし）")
            if result.stderr:
                st.text("STDERR:\n" + result.stderr)

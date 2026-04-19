# COLMAPを使ってカメラ姿勢推定（Structure from Motion）を実行するページ

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
frame_dirs = []
if experiments_dir.exists():
    frame_dirs = sorted([
        str(p) for p in experiments_dir.glob("*/frames") if p.is_dir() and any(p.iterdir())
    ])

col1, col2 = st.columns(2)

with col1:
    if frame_dirs:
        selected_frames = st.selectbox("フレームフォルダ（experiments/配下）", frame_dirs)
    else:
        st.warning("experiments/ 配下にフレームフォルダが見つかりません。先にフレーム抽出を実行してください。")
        selected_frames = st.text_input("フレームフォルダのパスを直接入力",
                                        placeholder="/workspace/experiments/20240101_120000_scene1/frames")

with col2:
    camera_model = st.selectbox("カメラモデル", ["OPENCV", "PINHOLE", "SIMPLE_RADIAL"],
                                 help="通常はOPENCVを推奨。360度変換済み画像はPINHOLE。")

# ── 出力設定 ──────────────────────────────────────────────────────────────────
st.subheader("出力設定")

if selected_frames:
    experiment_dir = str(Path(selected_frames).parent)
    default_output = f"{experiment_dir}/colmap"
else:
    default_output = ""

output_path = st.text_input("COLMAPの出力フォルダ", value=default_output)

# ── 詳細設定 ──────────────────────────────────────────────────────────────────
with st.expander("詳細設定"):
    use_gpu = st.checkbox("GPU使用（SIFT特徴点抽出）", value=True)
    single_camera = st.checkbox("全画像を同一カメラとみなす", value=True,
                                help="同じカメラで撮影した場合はONを推奨")

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")

cmd_args = [
    f"python /workspace/scripts/run_colmap.py",
    f"--image_path \"{selected_frames}\"",
    f"--output_path \"{output_path}\"",
    f"--camera_model {camera_model}",
]
if use_gpu:
    cmd_args.append("--use_gpu")
if single_camera:
    cmd_args.append("--single_camera")

st.code(" \\\n  ".join(cmd_args), language="bash")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()

st.warning("⚠️ COLMAPはGPUを使用します。処理時間はフレーム数によって数分〜数十分かかります。")

if st.button("▶ COLMAPを実行", type="primary", disabled=not selected_frames):
    if not os.path.exists(selected_frames):
        st.error(f"フォルダが見つかりません: {selected_frames}")
    else:
        os.makedirs(output_path, exist_ok=True)
        st.info("COLMAPを実行中です。しばらくお待ちください...")

        run_args = ["python", "/workspace/scripts/run_colmap.py",
                    "--image_path", selected_frames,
                    "--output_path", output_path,
                    "--camera_model", camera_model]
        if use_gpu:
            run_args.append("--use_gpu")
        if single_camera:
            run_args.append("--single_camera")

        result = subprocess.run(run_args, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("COLMAP が正常に完了しました！")
            sparse_dir = Path(output_path) / "sparse"
            if sparse_dir.exists():
                st.metric("sparse モデル数", len(list(sparse_dir.iterdir())))
        else:
            st.error("COLMAPの実行中にエラーが発生しました。")

        with st.expander("ログを表示"):
            st.text(result.stdout or "（出力なし）")
            if result.stderr:
                st.text("STDERR:\n" + result.stderr)

# 動画ファイルからFFmpegを使って連番画像（フレーム）を切り出すページ

import streamlit as st
import subprocess
import os
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="フレーム抽出", page_icon="🎞️", layout="wide")

st.title("🎞️ フレーム抽出")
st.caption("動画ファイルから連番画像を切り出します（FFmpeg使用）")

st.divider()

# ── 入力設定 ──────────────────────────────────────────────────────────────────
st.subheader("入力設定")

data_dir = Path("/workspace/data")
video_files = []
if data_dir.exists():
    video_files = sorted([
        str(p.relative_to("/workspace")) for p in data_dir.rglob("*.mp4")
    ] + [
        str(p.relative_to("/workspace")) for p in data_dir.rglob("*.mov")
    ] + [
        str(p.relative_to("/workspace")) for p in data_dir.rglob("*.avi")
    ])

col1, col2 = st.columns(2)

with col1:
    if video_files:
        selected_video = st.selectbox("動画ファイル（data/配下）", video_files)
        input_path = f"/workspace/{selected_video}"
    else:
        st.warning("data/ 配下に動画ファイルが見つかりません。")
        input_path = st.text_input("動画ファイルのパスを直接入力", placeholder="/workspace/data/scene1/video.mp4")

with col2:
    fps = st.number_input("抽出FPS（フレーム/秒）", min_value=0.1, max_value=60.0, value=2.0, step=0.5,
                          help="1なら1秒に1枚、2なら1秒に2枚抽出します")

# ── 出力設定 ──────────────────────────────────────────────────────────────────
st.subheader("出力設定")

if input_path:
    scene_name = Path(input_path).parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"/workspace/experiments/{timestamp}_{scene_name}/frames"
else:
    default_output = "/workspace/experiments/"

output_path = st.text_input("出力フォルダ", value=default_output)

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")

cmd = f"python /workspace/scripts/extract_frames.py --input \"{input_path}\" --output \"{output_path}\" --fps {fps}"
st.code(cmd, language="bash")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()

if st.button("▶ フレーム抽出を開始", type="primary", disabled=not input_path):
    if not os.path.exists(input_path):
        st.error(f"ファイルが見つかりません: {input_path}")
    else:
        os.makedirs(output_path, exist_ok=True)
        st.info(f"抽出中... 出力先: `{output_path}`")

        log_area = st.empty()
        result = subprocess.run(
            ["python", "/workspace/scripts/extract_frames.py",
             "--input", input_path,
             "--output", output_path,
             "--fps", str(fps)],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            st.success("フレーム抽出が完了しました！")
            frames = list(Path(output_path).glob("*.jpg")) + list(Path(output_path).glob("*.png"))
            st.metric("抽出枚数", f"{len(frames)} 枚")
        else:
            st.error("エラーが発生しました。")

        with st.expander("ログを表示"):
            st.text(result.stdout or "（出力なし）")
            if result.stderr:
                st.text("STDERR:\n" + result.stderr)

# 動画ファイルからFFmpegを使って連番画像（フレーム）を切り出すページ
# 抽出前のプレビュー機能付き。出力先は experiment/input/（gaussian-splatting標準構成）

import streamlit as st
import subprocess
import os
import tempfile
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
    for subdir in ("360movies", "movies"):
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            video_files += [str(p.relative_to("/workspace")) for p in (data_dir / subdir).rglob(ext)]
    video_files = sorted(video_files)

col1, col2 = st.columns(2)

with col1:
    if video_files:
        selected_video = st.selectbox("動画ファイル（data/360movies/ または data/movies/）", video_files)
        input_path = f"/workspace/{selected_video}"
    else:
        st.warning("data/360movies/ または data/movies/ に動画ファイルが見つかりません。")
        input_path = st.text_input("動画ファイルのパスを直接入力",
                                   placeholder="/workspace/data/movies/scene1.mp4")

with col2:
    fps = st.number_input("抽出FPS（フレーム/秒）", min_value=0.1, max_value=60.0,
                          value=2.0, step=0.5,
                          help="1なら1秒に1枚、2なら1秒に2枚。多すぎると似た画像が増えるので注意。")

# ── 動画情報の取得 ────────────────────────────────────────────────────────────
video_info = {}
if input_path and os.path.exists(input_path):
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=0", input_path],
            capture_output=True, text=True, timeout=10,
        )
        for line in probe.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                video_info[k.strip()] = v.strip()
    except Exception:
        pass

if video_info:
    duration = float(video_info.get("duration", 0))
    width = video_info.get("width", "?")
    height = video_info.get("height", "?")
    estimated = int(duration * fps)

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("解像度", f"{width}×{height}")
    ic2.metric("動画の長さ", f"{duration:.1f} 秒")
    ic3.metric("抽出予定枚数", f"{estimated} 枚")
    ic4.metric("元FPS（参考）", video_info.get("r_frame_rate", "?"))

# ── プレビュー ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📸 抽出プレビュー（5点サンプル）")
st.caption("本抽出前に代表フレームを確認できます。FPSが適切かチェックしてください。")

if st.button("🔍 プレビューを表示", disabled=not (input_path and os.path.exists(input_path))):
    duration = float(video_info.get("duration", 0)) if video_info else 0
    if duration <= 0:
        st.warning("動画の長さを取得できませんでした。")
    else:
        preview_dir = Path("/workspace/tmp/preview")
        preview_dir.mkdir(parents=True, exist_ok=True)

        times = [duration * t for t in [0.05, 0.25, 0.5, 0.75, 0.95]]
        preview_imgs = []

        with st.spinner("プレビュー画像を生成中..."):
            for i, t in enumerate(times):
                out_path = str(preview_dir / f"prev_{i}.jpg")
                subprocess.run(
                    ["ffmpeg", "-ss", str(t), "-i", input_path,
                     "-vframes", "1", "-q:v", "5", out_path, "-y"],
                    capture_output=True,
                )
                if os.path.exists(out_path):
                    preview_imgs.append((out_path, f"{t:.1f}s"))

        if preview_imgs:
            cols = st.columns(len(preview_imgs))
            for col, (img_path, label) in zip(cols, preview_imgs):
                col.image(img_path, caption=label, use_container_width=True)
        else:
            st.error("プレビュー画像の生成に失敗しました。")

# ── 出力設定 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("出力設定")

if input_path:
    scene_name = Path(input_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_exp = f"/workspace/experiments/{timestamp}_{scene_name}"
else:
    default_exp = "/workspace/experiments/"

experiment_dir = st.text_input("実験フォルダ", value=default_exp,
                               help="この下に input/ フォルダが作られます")
output_path = str(Path(experiment_dir) / "input") if experiment_dir else ""

if output_path:
    st.caption(f"フレームの保存先: `{output_path}`")

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")
cmd_str = (f'python /workspace/scripts/extract_frames.py '
           f'--input "{input_path}" --output "{output_path}" --fps {fps}')
st.code(cmd_str, language="bash")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()

if st.button("▶ フレーム抽出を開始", type="primary",
             disabled=not (input_path and output_path)):
    if not os.path.exists(input_path):
        st.error(f"ファイルが見つかりません: {input_path}")
    else:
        os.makedirs(output_path, exist_ok=True)
        st.info(f"抽出中... 出力先: `{output_path}`")

        with st.spinner("FFmpegで抽出中..."):
            result = subprocess.run(
                ["python", "/workspace/scripts/extract_frames.py",
                 "--input", input_path,
                 "--output", output_path,
                 "--fps", str(fps)],
                capture_output=True, text=True,
            )

        if result.returncode == 0:
            frames = (list(Path(output_path).glob("*.jpg")) +
                      list(Path(output_path).glob("*.png")))
            st.success(f"フレーム抽出完了！ {len(frames)} 枚")
            st.metric("抽出枚数", f"{len(frames)} 枚")
            st.info(f"次のステップ：「📷 COLMAP実行」ページで `{experiment_dir}` を選択してください。")
        else:
            st.error("エラーが発生しました。")

        with st.expander("ログを表示"):
            st.text(result.stdout or "（出力なし）")
            if result.stderr:
                st.text("STDERR:\n" + result.stderr)

# 360度（等距円筒）動画・画像をピンホールカメラ視点の画像群に変換するページ
# フレーム抽出の前段（Step 0）として使用する

import os
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="360度変換", page_icon="🌐", layout="wide")

st.title("🌐 360度動画変換（Step 0）")
st.caption("360度動画・画像をピンホールカメラ視点の画像群に変換します")

st.info(
    "**使い方の流れ**\n"
    "1. このページで360度動画 → ピンホール画像群に変換\n"
    "2. 変換後の画像を `data/images/<シーン名>/` に保存\n"
    "3. 「フレーム抽出」ページはスキップして「COLMAP実行」ページへ進む"
)

st.divider()

# ── 入力設定 ──────────────────────────────────────────────────────────────────
st.subheader("入力設定")

data_360_dir = Path("/workspace/data/360movies")
video_files = []
if data_360_dir.exists():
    for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        video_files += [str(p.relative_to("/workspace")) for p in data_360_dir.rglob(ext)]
video_files = sorted(video_files)

col1, col2 = st.columns(2)

with col1:
    input_type = st.radio("入力タイプ", ["動画ファイル（data/360movies/）", "画像フォルダを直接指定"])

with col2:
    if input_type == "動画ファイル（data/360movies/）":
        if video_files:
            sel = st.selectbox("360度動画ファイル", video_files)
            input_path = f"/workspace/{sel}"
        else:
            st.warning("data/360movies/ に動画が見つかりません。")
            input_path = st.text_input("パスを直接入力",
                                       placeholder="/workspace/data/360movies/scene.mp4")
    else:
        input_path = st.text_input("画像フォルダのパスを入力",
                                   placeholder="/workspace/data/images/scene_equirect/")

# ── 変換設定 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("変換設定")

col3, col4, col5 = st.columns(3)

with col3:
    fov = st.slider("水平視野角（FOV）", min_value=60, max_value=120, value=90, step=5,
                    help="小さいほど望遠、大きいほど広角。90度が標準的。")

with col4:
    out_size = st.selectbox("出力解像度", ["512×512", "1024×1024", "2048×2048"], index=1)
    out_w = out_h = int(out_size.split("×")[0])

with col5:
    fps = st.number_input("抽出FPS（動画入力時）", min_value=0.1, max_value=10.0,
                          value=1.0, step=0.5,
                          help="1なら1秒に1フレーム。多くしすぎると画像枚数が増えすぎます。")

direction_options = ["front", "right", "back", "left", "up", "down"]
direction_labels = {
    "front": "前（front）",
    "right": "右（right）",
    "back": "後（back）",
    "left": "左（left）",
    "up": "上（up）",
    "down": "下（down）",
}
selected_dirs = st.multiselect(
    "変換する方向",
    direction_options,
    default=["front", "right", "back", "left"],
    format_func=lambda x: direction_labels[x],
    help="選んだ方向ごとに1枚ずつ画像が生成されます。",
)

# ── 出力先設定 ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("出力先設定")

scene_name = Path(input_path).stem if input_path else "scene"
default_out = f"/workspace/data/images/{scene_name}_pinhole"
output_path = st.text_input("出力フォルダ", value=default_out,
                             help="変換後の画像はここに保存されます。COLMAP実行時にこのフォルダを指定してください。")

if input_path and selected_dirs:
    est_frames = "不明"
    if input_path and os.path.exists(input_path) and Path(input_path).is_file():
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", input_path],
                capture_output=True, text=True, timeout=10,
            )
            dur = float(probe.stdout.strip())
            est_frames = int(dur * fps) * len(selected_dirs)
        except Exception:
            pass
    st.caption(f"出力予定枚数の目安: {est_frames} 枚（{len(selected_dirs)}方向 × フレーム数）")

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")
cmd_str = (
    f'python /workspace/scripts/convert_360.py \\\n'
    f'  --input "{input_path}" \\\n'
    f'  --output "{output_path}" \\\n'
    f'  --fov {fov} --width {out_w} --height {out_h} \\\n'
    f'  --fps {fps} \\\n'
    f'  --directions {" ".join(selected_dirs)}'
)
st.code(cmd_str, language="bash")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()

can_run = bool(input_path and output_path and selected_dirs)

if st.button("▶ 変換を開始", type="primary", disabled=not can_run):
    if not os.path.exists(input_path):
        st.error(f"入力が見つかりません: {input_path}")
    else:
        os.makedirs(output_path, exist_ok=True)
        cmd = [
            "python", "/workspace/scripts/convert_360.py",
            "--input", input_path,
            "--output", output_path,
            "--fov", str(fov),
            "--width", str(out_w),
            "--height", str(out_h),
            "--fps", str(fps),
            "--directions", *selected_dirs,
        ]
        st.info(f"変換中... 出力先: `{output_path}`")
        with st.spinner("変換処理中（動画が長いと数分かかります）..."):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            n = len(list(Path(output_path).glob("*.jpg")))
            st.success(f"変換完了！ {n} 枚の画像を生成しました。")
            st.info(
                f"次のステップ：「📷 COLMAP実行」ページで\n"
                f"実験フォルダに `{output_path}` を `input/` としてシンボリックリンクするか、\n"
                f"直接コピーして使用してください。"
            )
        else:
            st.error("変換中にエラーが発生しました。")

        with st.expander("ログを表示"):
            st.text(result.stdout or "（出力なし）")
            if result.stderr:
                st.text("STDERR:\n" + result.stderr)

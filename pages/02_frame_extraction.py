# 動画ファイルから連番画像を切り出すページ（360度動画のピンホール変換オプション付き）
# 通常動画はFFmpegでフレーム抽出、360度動画はピンホール変換を行ってから出力する

import os
import re
import signal
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

import streamlit as st


# ── セッション状態の初期化 ────────────────────────────────────────────────────
if "extract_proc" not in st.session_state:
    st.session_state.extract_proc     = None
    st.session_state.extract_log_path = None
    st.session_state.extract_scene    = None
    st.session_state.extract_is_360   = False


def _extract_running():
    p = st.session_state.extract_proc
    return p is not None and p.poll() is None


# ── 実行ステータスバナー（常時表示） ──────────────────────────────────────────
_proc = st.session_state.extract_proc
if _proc is not None:
    _running = _proc.poll() is None
    if _running:
        _ba, _bb = st.columns([5, 1])
        _ba.info(f"🔄 フレーム抽出実行中　｜　詳細は「🗂️ バッチキュー」ページで確認できます")
        if _bb.button("⏹ 中断", key="fe_stop"):
            try: os.kill(_proc.pid, signal.SIGTERM)
            except Exception: pass
            st.session_state.extract_proc = None
            st.session_state.pop("active_task", None)
            sys.path.insert(0, "/workspace")
            from queue_helper import clear_active_task_file as _clf; _clf()
            st.rerun()
        time.sleep(2)
        st.rerun()
    elif _proc.returncode == 0:
        st.success("✅ フレーム抽出完了！次は「📷 姿勢推定」ページへ。")
        if st.button("✕ クリア", key="fe_clear"):
            st.session_state.extract_proc = None
            st.rerun()
    else:
        st.error(f"❌ エラーで終了しました（終了コード: {_proc.returncode}）")
        if st.button("✕ クリア", key="fe_clear_err"):
            st.session_state.extract_proc = None
            st.rerun()

st.title("🎞️ フレーム抽出")
st.caption("動画ファイルから画像を切り出します。360度動画の場合はピンホール変換オプションを使用してください。")

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

if video_files:
    selected_video = st.selectbox("動画ファイル（data/360movies/ または data/movies/）", video_files)
    input_path = f"/workspace/{selected_video}"
else:
    st.warning("data/360movies/ または data/movies/ に動画ファイルが見つかりません。")
    input_path = st.text_input("動画ファイルのパスを直接入力",
                               placeholder="/workspace/data/movies/scene1.mp4")

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

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("解像度", f"{width}×{height}")
    ic2.metric("動画の長さ", f"{duration:.1f} 秒")
    ic3.metric("元FPS（参考）", video_info.get("r_frame_rate", "?"))

st.divider()

# ── 360度変換オプション ───────────────────────────────────────────────────────
is_360 = st.checkbox(
    "360度動画（ピンホール変換する）",
    help="等距円筒（Equirectangular）形式の360度動画の場合にチェックしてください。"
         "各フレームを複数のカメラ視点画像に変換します。",
)

sel_angles = []

if is_360:
    st.info("各フレームを指定した方向のピンホールカメラ視点に変換します。")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fov = st.slider("水平視野角（FOV）", min_value=60, max_value=120, value=90, step=5,
                        help="小さいほど望遠、大きいほど広角。90度が標準的。")
    with col_b:
        out_size = st.selectbox("出力解像度", ["512×512", "1024×1024", "2048×2048"], index=1)
        out_w = out_h = int(out_size.split("×")[0])
    with col_c:
        fps = st.number_input("抽出FPS", min_value=0.1, max_value=10.0, value=1.0, step=0.5,
                              help="360度動画は1枚あたり複数画像が生成されるため、低めのFPSを推奨します。")

    # 8×3 方向選択グリッド（水平45°刻み × 垂直-30°/0°/+30°）
    st.markdown("**変換する方向を選択（水平角 × 垂直角）**")
    YAW_ANGLES  = [0, 45, 90, 135, 180, 225, 270, 315]
    PITCH_ANGLES = [30, 0, -30]
    YAW_SHORT   = ["前\n0°", "45°", "右\n90°", "135°", "後\n180°", "225°", "左\n270°", "315°"]
    PITCH_LABELS = {30: "上 +30°", 0: "水平  0°", -30: "下 −30°"}

    header_cols = st.columns([1.5] + [1] * 8)
    header_cols[0].write("")
    for i, label in enumerate(YAW_SHORT):
        header_cols[i + 1].markdown(label)

    angle_checks = {}
    for pitch in PITCH_ANGLES:
        row_cols = st.columns([1.5] + [1] * 8)
        row_cols[0].markdown(f"**{PITCH_LABELS[pitch]}**")
        for i, yaw in enumerate(YAW_ANGLES):
            angle_checks[(yaw, pitch)] = row_cols[i + 1].checkbox(
                "　",
                value=(pitch == 0),
                key=f"fe_360_y{yaw}_p{pitch}",
                label_visibility="collapsed",
            )

    sel_angles = [(y, p) for (y, p), v in angle_checks.items() if v]

    if video_info and sel_angles:
        duration = float(video_info.get("duration", 0))
        est = int(duration * fps) * len(sel_angles)
        st.caption(f"出力予定枚数の目安: {est} 枚（{len(sel_angles)}方向 × {int(duration * fps)}フレーム）")
else:
    fov, out_w, out_h = 90, 1024, 1024  # 未使用だが変数として定義しておく
    fps = st.number_input("抽出FPS（フレーム/秒）", min_value=0.1, max_value=60.0,
                          value=2.0, step=0.5,
                          help="1なら1秒に1枚、2なら1秒に2枚。多すぎると似た画像が増えるので注意。")
    if video_info:
        duration = float(video_info.get("duration", 0))
        st.caption(f"抽出予定枚数の目安: {int(duration * fps)} 枚")

# ── プレビュー（通常モードのみ） ──────────────────────────────────────────────
if not is_360:
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
    import sys as _sys2; _sys2.path.insert(0, "/workspace")
    from queue_helper import next_exp_name as _nxt
    default_exp = f"/workspace/experiments/{_nxt(scene_name)}"
else:
    default_exp = "/workspace/experiments/"

experiment_dir = st.text_input("実験フォルダ", value=default_exp,
                               help="この下に input/ フォルダが作られます")
output_path = str(Path(experiment_dir) / "input") if experiment_dir else ""

if output_path:
    st.caption(f"画像の保存先: `{output_path}`")

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")
if is_360 and sel_angles:
    angles_str = " ".join(f"{y},{p}" for y, p in sel_angles)
    cmd_str = (
        f'python /workspace/scripts/convert_360.py \\\n'
        f'  --input "{input_path}" --output "{output_path}" \\\n'
        f'  --fov {fov} --width {out_w} --height {out_h} \\\n'
        f'  --fps {fps} --angles {angles_str}'
    )
else:
    cmd_str = (f'python /workspace/scripts/extract_frames.py '
               f'--input "{input_path}" --output "{output_path}" --fps {fps}')
st.code(cmd_str, language="bash")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()

can_run = bool(input_path and output_path and (not is_360 or sel_angles))

if st.button("▶ 抽出を開始", type="primary", disabled=not can_run):
    if not os.path.exists(input_path):
        st.error(f"ファイルが見つかりません: {input_path}")
    else:
        os.makedirs(output_path, exist_ok=True)
        log_path = str(Path(experiment_dir) / "extract_log.txt")

        if is_360:
            cmd = [
                sys.executable, "/workspace/scripts/convert_360.py",
                "--input", input_path,
                "--output", output_path,
                "--fov", str(fov),
                "--width", str(out_w),
                "--height", str(out_h),
                "--fps", str(fps),
                "--angles", *[f"{y},{p}" for y, p in sel_angles],
            ]
        else:
            cmd = [
                sys.executable, "/workspace/scripts/extract_frames.py",
                "--input", input_path,
                "--output", output_path,
                "--fps", str(fps),
            ]

        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        scene = Path(experiment_dir).name
        st.session_state.extract_proc     = proc
        st.session_state.extract_log_path = log_path
        st.session_state.extract_scene    = scene
        st.session_state.extract_is_360   = is_360
        st.session_state.active_task = {
            "step":       "extracting",
            "label":      "フレーム抽出",
            "scene":      scene,
            "log_path":   log_path,
            "pid":        proc.pid,
            "start_time": time.time(),
            "is_360":     is_360,
        }
        from queue_helper import save_active_task_file as _satf
        _satf(st.session_state.active_task)
        st.rerun()

# ── キューに追加 ──────────────────────────────────────────────────────────────
sys.path.insert(0, "/workspace")
from queue_helper import add_to_queue, pending_size as _psize

if st.button(
    f"📋 バッチキューに追加（待ち: {_psize()} 件）",
    disabled=not can_run,
    use_container_width=True,
):
    if not os.path.exists(input_path):
        st.error(f"ファイルが見つかりません: {input_path}")
    else:
        add_to_queue(
            job_type="extract",
            label=f"フレーム抽出 {fps}fps",
            exp_name=Path(experiment_dir).name,
            exp_dir=experiment_dir,
            config={
                "video_path": input_path,
                "fps": float(fps),
                "is_360": is_360,
                "fov": fov, "out_w": out_w, "out_h": out_h,
                "angles": sel_angles,
            },
        )
        st.success("バッチキューに追加しました。")

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### FPS（フレームレート）の目安

| FPS | 1分動画の枚数 | 推奨シーン |
|---|---|---|
| 1 | 60枚 | 手持ち撮影・ゆっくり動かす場合 |
| 2 | 120枚 | 標準（デフォルト） |
| 5 | 300枚 | 高密度・細部重視 |
| 10以上 | 600枚〜 | 非推奨（COLMAPが遅くなる） |

枚数が多いほどCOLMAPの処理時間が増加します。**100〜500枚程度** を目安にしてください。

---

### 通常動画 vs 360度動画

| 種別 | 処理内容 |
|---|---|
| **通常動画** | FFmpegでフレーム切り出しのみ |
| **360度動画** | 等距円筒→ピンホール変換（複数方向に投影）してからCOLMAPへ |

360度動画の場合は **撮影方向（Yaw/Pitch）** と **解像度** を指定してください。
方向を多く指定するほど点群が密になりますが処理時間も増加します。

---

### 出力先フォルダ
- `experiments/<日時>_<シーン名>/input/` に連番画像が保存されます。
- フレーム抽出後は「📷 姿勢推定」ページへ進んでください。
""")